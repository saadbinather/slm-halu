import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

FEATURE_DIR = Path('processed_features')
MODEL_DIR = Path('models')
RESULT_DIR = Path('results')
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_files(split):
    folder = FEATURE_DIR / split
    if not folder.exists():
        raise FileNotFoundError(f'Missing folder: {folder}')
    files = sorted(folder.glob('*.pt'))
    if not files:
        raise RuntimeError(f'No .pt files found in {folder}')
    return files


def load_trace(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def find_best_threshold(y_true, probs):
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        preds = (probs >= threshold).astype(np.int64)
        score = balanced_accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics(y_true, probs, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    preds = (probs >= threshold).astype(np.int64)
    return {
        'threshold': float(threshold),
        'auroc': float(roc_auc_score(y_true, probs)),
        'auprc_hallucinated': float(average_precision_score(y_true, probs)),
        'auprc_correct': float(average_precision_score(1 - y_true, 1.0 - probs)),
        'macro_f1': float(f1_score(y_true, preds, average='macro')),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, preds)),
        'accuracy': float(accuracy_score(y_true, preds)),
    }


def print_metrics(title, metrics):
    print('\n' + title)
    print('-' * len(title))
    print(f"AUROC:              {metrics['auroc']:.4f}")
    print(f"AUPRC hallucinated: {metrics['auprc_hallucinated']:.4f}")
    print(f"AUPRC correct:      {metrics['auprc_correct']:.4f}")
    print(f"Macro F1:           {metrics['macro_f1']:.4f}")
    print(f"Balanced accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"Threshold:           {metrics['threshold']:.3f}")


# =========================================================
# BASELINE 1: SINGLE HIDDEN-STATE LOGISTIC REGRESSION
# Uses t=0 and the final transformer layer only.
# =========================================================

def build_single_state_dataset(split):
    X, y, sample_ids = [], [], []
    for path in tqdm(get_files(split), desc=f'Single-state {split}'):
        trace = load_trace(path)
        label = int(trace['label'])
        if label not in [0, 1]:
            continue
        features = trace['features'].float()  # [T, L, 16]
        times = list(trace['time_positions'])
        if 0 not in times:
            raise RuntimeError(f'{path.name} has no t=0 state.')
        t0_idx = times.index(0)
        vector = features[t0_idx, -1, :].numpy()  # final layer, t=0
        X.append(vector)
        y.append(label)
        sample_ids.append(str(trace['sample_id']))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), sample_ids


def run_logistic_regression():
    print('\n' + '=' * 70)
    print('BASELINE 1 — SINGLE-STATE LOGISTIC REGRESSION')
    print('=' * 70)

    X_train, y_train, _ = build_single_state_dataset('train')
    X_val, y_val, _ = build_single_state_dataset('val')
    X_test, y_test, _ = build_single_state_dataset('test')

    print(f'\nTrain shape: {X_train.shape}')
    print(f'Val shape:   {X_val.shape}')
    print(f'Test shape:  {X_test.shape}')
    print(f'Train labels -> correct={np.sum(y_train == 0)}, hallucinated={np.sum(y_train == 1)}')

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=5000, random_state=SEED)),
    ])

    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    threshold = find_best_threshold(y_val, val_probs)
    val_metrics = compute_metrics(y_val, val_probs, threshold)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_probs, threshold)

    print_metrics('Validation', val_metrics)
    print_metrics('Test', test_metrics)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / 'baseline_single_state_logreg.joblib'
    joblib.dump(model, model_path)

    return {
        'name': 'single_state_logistic_regression',
        'input': 't=0, final transformer layer, PCA-16',
        'validation': val_metrics,
        'test': test_metrics,
        'model_path': str(model_path),
    }


# =========================================================
# BASELINE 2: FLATTENED TRAJECTORY MLP
# Canonical time slots are inferred from TRAIN only.
# Missing time slots are zero-padded; a presence mask is appended.
# No graph edges are used.
# =========================================================

def infer_train_time_positions():
    times = set()
    for path in tqdm(get_files('train'), desc='Finding train time positions'):
        trace = load_trace(path)
        for t in trace['time_positions']:
            times.add(int(t))
    canonical_times = sorted(times)
    if not canonical_times:
        raise RuntimeError('Could not infer any time positions.')
    print(f'\nCanonical TRAIN time positions: {canonical_times}')
    return canonical_times


def trajectory_to_vector(trace, canonical_times):
    features = trace['features'].float()
    num_layers = features.shape[1]
    pca_dim = features.shape[2]

    padded = torch.zeros(len(canonical_times), num_layers, pca_dim, dtype=torch.float32)
    mask = torch.zeros(len(canonical_times), dtype=torch.float32)
    time_to_slot = {int(t): i for i, t in enumerate(canonical_times)}

    for local_idx, t in enumerate(trace['time_positions']):
        t = int(t)
        if t not in time_to_slot:
            continue
        slot = time_to_slot[t]
        padded[slot] = features[local_idx]
        mask[slot] = 1.0

    return torch.cat([padded.reshape(-1), mask], dim=0).numpy()


def build_trajectory_dataset(split, canonical_times):
    X, y, sample_ids = [], [], []
    for path in tqdm(get_files(split), desc=f'Trajectory {split}'):
        trace = load_trace(path)
        label = int(trace['label'])
        if label not in [0, 1]:
            continue
        X.append(trajectory_to_vector(trace, canonical_times))
        y.append(label)
        sample_ids.append(str(trace['sample_id']))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), sample_ids


class TrajectoryMLP(nn.Module):
    def __init__(self, input_dim, hidden1=256, hidden2=64, dropout=0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def predict_mlp(model, X, device, batch_size):
    model.eval()
    tensor_x = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_x), batch_size=batch_size, shuffle=False)
    probs = []
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def run_trajectory_mlp(epochs, batch_size, learning_rate, patience):
    print('\n' + '=' * 70)
    print('BASELINE 2 — FLATTENED TRAJECTORY MLP')
    print('=' * 70)

    canonical_times = infer_train_time_positions()
    X_train, y_train, _ = build_trajectory_dataset('train', canonical_times)
    X_val, y_val, _ = build_trajectory_dataset('val', canonical_times)
    X_test, y_test, _ = build_trajectory_dataset('test', canonical_times)

    print(f'\nTrain shape: {X_train.shape}')
    print(f'Val shape:   {X_val.shape}')
    print(f'Test shape:  {X_test.shape}')
    print(f'Train labels -> correct={np.sum(y_train == 0)}, hallucinated={np.sum(y_train == 1)}')

    # Fit scaler on TRAIN only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')

    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    model = TrajectoryMLP(input_dim=X_train.shape[1]).to(device)

    num_negative = float(np.sum(y_train == 0))
    num_positive = float(np.sum(y_train == 1))
    pos_weight_value = num_negative / max(num_positive, 1.0)
    print(f'Positive-class weight: {pos_weight_value:.4f}')

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODEL_DIR / 'baseline_flat_trajectory_mlp.pt'

    best_val_auroc = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            n = batch_x.shape[0]
            running_loss += loss.item() * n
            seen += n

        train_loss = running_loss / max(seen, 1)
        val_probs = predict_mlp(model, X_val, device, batch_size)
        val_auroc = roc_auc_score(y_val, val_probs)

        print(f'Epoch {epoch:03d} | loss={train_loss:.4f} | val_AUROC={val_auroc:.4f}')

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_without_improvement = 0
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'input_dim': int(X_train.shape[1]),
                    'canonical_times': canonical_times,
                    'best_val_auroc': float(best_val_auroc),
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'\nEarly stopping after {epoch} epochs.')
            break

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    val_probs = predict_mlp(model, X_val, device, batch_size)
    threshold = find_best_threshold(y_val, val_probs)
    val_metrics = compute_metrics(y_val, val_probs, threshold)

    test_probs = predict_mlp(model, X_test, device, batch_size)
    test_metrics = compute_metrics(y_test, test_probs, threshold)

    print_metrics('Validation', val_metrics)
    print_metrics('Test', test_metrics)

    scaler_path = MODEL_DIR / 'baseline_flat_trajectory_scaler.joblib'
    joblib.dump(scaler, scaler_path)

    return {
        'name': 'flattened_trajectory_mlp',
        'canonical_times': canonical_times,
        'input_dim': int(X_train.shape[1]),
        'validation': val_metrics,
        'test': test_metrics,
        'model_path': str(best_model_path),
        'scaler_path': str(scaler_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('EPIGRAPH — STEP 07: GRAPH-FREE BASELINES')
    print('=' * 70)

    logistic_results = run_logistic_regression()
    mlp_results = run_trajectory_mlp(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
    )

    results = {
        'seed': args.seed,
        'label_definition': {'0': 'correct', '1': 'hallucinated'},
        'single_state_logistic_regression': logistic_results,
        'flattened_trajectory_mlp': mlp_results,
    }

    result_path = RESULT_DIR / 'step07_baselines.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print('\n' + '=' * 70)
    print('STEP 07 COMPLETE')
    print('=' * 70)
    print(f'\nResults saved to:\n{result_path}')
    print('\nNext step: train EpiGraph GNN on the graph files.')


if __name__ == '__main__':
    main()
