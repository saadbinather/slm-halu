# SLM-Halu: GNN-Based Hallucination Detection in Small Language Models

**SLM-Halu** is a research framework for detecting hallucinations in Small Language Models (SLMs)—specifically **Qwen2.5-3B-Instruct**—by modeling layer-wise internal activation traces as sequential Graph Neural Networks (GNNs).

Rather than relying solely on black-box output text, this project extracts the LLM's hidden state representations across all transformer layers during generation, constructs layer-flow graphs, and uses Graph Convolutional Networks (GCNs) to classify responses as **Factual (0)** or **Hallucinated (1)**.

---

## 🏗️ Methodology Pipeline

1. **Trace Extraction (`qwen3b.py` / `movies.py`):**
   * Prompt the LLM on QA benchmarks (e.g., MovieQA, NQ, HaluEval).
   * Capture hidden states across all transformer layers ($N=37$) for generated outputs and save traces as `.pt` files.

2. **LLM Verification & Labeling (`label.py`):**
   * Apply exact string matching followed by an **LLM-as-a-Judge** (Qwen2.5-3B) verification step to label responses:
     * `0`: Factual / Correct Grounding
     * `1`: Hallucinated / Incorrect Information

3. **Graph Construction (`graphs.py`):**
   * Map transformer layers into graph nodes where node features $\mathbf{x}_i$ represent mean-pooled layer hidden states.
   * Connect sequential transformer layers via bidirectional edges $(0 \leftrightarrow 1 \leftrightarrow \dots \leftrightarrow N-1)$.

4. **GNN Classification (`train_gnn.py`):**
   * Train a 2-layer Graph Convolutional Network (`GCNConv`) with global mean pooling and class-weighted Cross-Entropy loss to handle data imbalances.

---

## 📁 Repository Structure

```text
slm-halu/
├── halueval_data/               # HaluEval dataset subsets (QA, Dialogue, General, Summarization)
├── movie_traces_train/          # Extracted .pt traces containing layer hidden states & labels
├── nq_data/                     # Natural Questions dataset artifacts
│
├── qwen3b.py                    # Base model loader & hidden state trace extractor
├── label.py                     # LLM-as-a-Judge automated ground-truth verifier
├── graphs.py                    # PyTorch Geometric Graph builder logic
│
├── train_gnn.py                 # Main GNN training pipeline (GCNConv + weighted CE loss)
├── test_movies.py               # Evaluation script for unseen test traces (AUROC & AUPR)
│
├── gnn_model.pt                 # Saved trained GNN weights
├── ds.py / sampleRes.py         # Data processing & sampling utility scripts
├── .gitignore                   # Git exclusion rules for large traces & virtualenvs
└── README.md                    # Project documentation