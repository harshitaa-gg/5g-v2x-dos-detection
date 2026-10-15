"""
src/train_transformer_extractor.py
==================================
Trains the Transformer Feature Extractor and extracts frozen 64-dimensional embeddings
for Highway 2 and Highway 7 (Option B).

Key Safeguards & Methodology:
-----------------------------
1. Data Leakage Prevention:
   - Training split ONLY is used for gradient updates.
   - Validation split is used strictly for model selection and early stopping.
   - Test split is strictly HELD OUT and never seen during training or checkpoint selection.
2. Early Stopping & Checkpoint Selection:
   - Evaluated on validation PR-AUC (with validation loss fallback).
   - Checkpoints:
     - `models/transformer/checkpoint_latest_{highway}.pt`: Full training state for resuming.
     - `models/transformer/transformer_best_{highway}.pt`: Best model weights based on val PR-AUC.
3. Resumability:
   - `--resume` flag resumes training from the latest checkpoint without restarting or overwriting best.
4. Embedding Extraction:
   - Freezes best model and extracts 64-d temporal mean-pooled embeddings for train, val, test.
   - Saved to `processed_data/transformer_embeddings/{highway}/{split}/embeddings.npz`.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Attempt sklearn metric import
try:
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import our Transformer model definition
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.models.transformer_encoder import TransformerFeatureExtractor

# ---------------------------------------------------------------------------
# Paths and Defaults
# ---------------------------------------------------------------------------
SEQ_BASE = os.path.join(PROJECT_ROOT, "processed_data", "sequences_tx")
CTX_BASE = os.path.join(PROJECT_ROOT, "processed_data", "transmission_context")
EMB_BASE = os.path.join(PROJECT_ROOT, "processed_data", "transformer_embeddings")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "transformer")

HIGHWAYS = ["highway_2", "highway_7"]
SPLITS = ["train", "validation", "test"]

DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 20
DEFAULT_PATIENCE = 5


def set_seed(seed: int = DEFAULT_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes PR-AUC, F1, precision, and recall.
    """
    metrics = {}
    if SKLEARN_AVAILABLE and len(np.unique(y_true)) > 1:
        try:
            prauc = float(average_precision_score(y_true, y_prob))
        except Exception:
            prauc = 0.0
    else:
        # Fallback if single class or sklearn missing
        prauc = 0.0

    preds = (y_prob >= 0.5).astype(np.int8)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    metrics["pr_auc"] = prauc
    metrics["f1"] = f1
    metrics["precision"] = prec
    metrics["recall"] = rec
    metrics["fpr"] = fpr
    return metrics


def load_dataset(highway: str, split: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Loads sequences from processed_data/sequences_tx/{highway}/{split}/sequences.npz.
    Returns: X_tensor, y_tensor, window_ids.
    """
    path = os.path.join(SEQ_BASE, highway, split, "sequences.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing sequence archive: {path}")

    npz = np.load(path)
    X = npz["X"]  # (N, 10, 9), float32
    y = npz["y"]  # (N,), int8
    w_ids = npz["window_ids"]  # (N,), int64

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    return X_tensor, y_tensor, w_ids


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_targets = []

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logits = model(X_b)
        loss = criterion(logits, y_b)
        probs = torch.sigmoid(logits)

        total_loss += loss.item()
        n_batches += 1
        all_probs.append(probs.cpu().numpy())
        all_targets.append(y_b.cpu().numpy().astype(np.int8))

    avg_loss = total_loss / max(1, n_batches)
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)
    metrics = compute_metrics(y_true, y_prob)
    metrics["val_loss"] = avg_loss
    return avg_loss, metrics


def train_highway(
    highway: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    patience: int = DEFAULT_PATIENCE,
    resume: bool = False,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """
    Trains the Transformer feature extractor for one highway scenario.
    """
    print(f"\n=================================================================")
    print(f"Training Transformer Feature Extractor: {highway}")
    print(f"=================================================================")
    set_seed(seed)
    device = torch.device("cpu")  # Strictly CPU-friendly execution

    os.makedirs(MODELS_DIR, exist_ok=True)
    latest_ckpt_path = os.path.join(MODELS_DIR, f"checkpoint_latest_{highway}.pt")
    best_ckpt_path = os.path.join(MODELS_DIR, f"transformer_best_{highway}.pt")

    # 1. Load train and validation splits
    print(f"  Loading train and validation datasets for {highway} ...")
    X_train, y_train, _ = load_dataset(highway, "train")
    X_val, y_val, _ = load_dataset(highway, "validation")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"  Train samples: {len(X_train):,}, Val samples: {len(X_val):,}")

    # 2. Initialize Model, Criterion, Optimizer, Scheduler
    model = TransformerFeatureExtractor().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    start_epoch = 1
    best_metric = -float("inf")
    patience_counter = 0
    history = []

    # 3. Check for resuming
    if resume and os.path.isfile(latest_ckpt_path):
        print(f"  Resuming from checkpoint: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", -float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)
        history = ckpt.get("history", [])
        print(f"  Resumed at epoch {start_epoch} (Best val PR-AUC: {best_metric:.4f})")

    total_training_start = time.time()

    # 4. Training Loop
    for epoch in range(start_epoch, epochs + 1):
        t_epoch = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)

        # Primary selection metric: PR-AUC (fallback to -val_loss if 0.0)
        current_metric = val_metrics["pr_auc"] if val_metrics["pr_auc"] > 0.0 else -val_loss
        scheduler.step(current_metric)

        epoch_time = time.time() - t_epoch

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_pr_auc": round(val_metrics["pr_auc"], 4),
            "val_f1": round(val_metrics["f1"], 4),
            "val_precision": round(val_metrics["precision"], 4),
            "val_recall": round(val_metrics["recall"], 4),
            "val_fpr": round(val_metrics["fpr"], 4),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "duration_s": round(epoch_time, 2),
        }
        history.append(log_entry)

        is_best = current_metric > best_metric
        status_flag = ""
        if is_best:
            best_metric = current_metric
            patience_counter = 0
            status_flag = " [BEST - Checkpoint Saved]"

            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_config": model.get_config(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "best_metric": best_metric,
                    "highway": highway,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                best_ckpt_path,
            )
        else:
            patience_counter += 1
            status_flag = f" (Patience: {patience_counter}/{patience})"

        # Save latest resumable checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": best_metric,
                "patience_counter": patience_counter,
                "history": history,
                "highway": highway,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            latest_ckpt_path,
        )

        print(
            f"  Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val PR-AUC: {val_metrics['pr_auc']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
            f"Val FPR: {val_metrics['fpr']*100:.2f}%{status_flag}"
        )

        if patience_counter >= patience:
            print(f"  Early stopping triggered after {patience} epochs without improvement.")
            break

    total_training_time = time.time() - total_training_start
    print(f"  Training finished in {total_training_time:.2f} s. Best model at: {best_ckpt_path}")

    # 5. Load the best model for embedding extraction
    print(f"\n  Loading best checkpoint for embedding extraction ...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # 6. Extract embeddings for all splits (train, validation, test)
    split_results = {}
    for split in SPLITS:
        print(f"  Extracting embeddings for {highway} / {split} ...")
        X_split, y_split, w_ids = load_dataset(highway, split)
        dataset_split = TensorDataset(X_split)
        loader_split = DataLoader(dataset_split, batch_size=batch_size, shuffle=False)

        all_embs = []
        with torch.no_grad():
            for (X_batch,) in loader_split:
                emb = model.extract_embedding(X_batch.to(device))  # (B, 64)
                all_embs.append(emb.cpu().numpy())

        emb_matrix = np.concatenate(all_embs, axis=0)  # (N, 64)
        assert emb_matrix.shape == (len(X_split), 64), f"Shape mismatch: {emb_matrix.shape}"
        assert not np.isnan(emb_matrix).any(), f"NaN in embeddings for {highway}/{split}!"
        assert not np.isinf(emb_matrix).any(), f"Inf in embeddings for {highway}/{split}!"

        # Save embeddings
        out_dir = os.path.join(EMB_BASE, highway, split)
        os.makedirs(out_dir, exist_ok=True)
        out_npz = os.path.join(out_dir, "embeddings.npz")
        np.savez_compressed(
            out_npz,
            embeddings=emb_matrix.astype(np.float32),
            window_ids=w_ids.astype(np.int64),
            labels=y_split.numpy().astype(np.int8),
        )
        file_sz_mb = os.path.getsize(out_npz) / (1024 * 1024)
        print(f"    Saved -> {out_npz} ({len(emb_matrix):,} rows, {file_sz_mb:.2f} MB)")

        # Verify 1-to-1 alignment with context features
        ctx_parquet_path = os.path.join(CTX_BASE, highway, split, "context_features.parquet")
        ctx_df = pq.read_table(ctx_parquet_path, columns=["window_id", "label"]).to_pandas()
        assert len(ctx_df) == len(emb_matrix), "Count mismatch with context features!"
        assert (ctx_df["window_id"].to_numpy() == w_ids).all(), "window_id alignment error!"
        assert (ctx_df["label"].to_numpy() == y_split.numpy()).all(), "label alignment error!"

        # Save meta.json
        meta_json_path = os.path.join(out_dir, "meta.json")
        split_meta = {
            "highway": highway,
            "split": split,
            "n_samples": len(emb_matrix),
            "embedding_dim": 64,
            "dtype": "float32",
            "source_checkpoint": best_ckpt_path,
            "best_epoch": best_ckpt["epoch"],
            "alignment_verified": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_json_path, "w") as fp:
            json.dump(split_meta, fp, indent=2)

        split_results[split] = split_meta

    return {
        "highway": highway,
        "best_epoch": best_ckpt["epoch"],
        "best_val_pr_auc": best_ckpt["val_metrics"]["pr_auc"],
        "best_val_f1": best_ckpt["val_metrics"]["f1"],
        "best_val_loss": best_ckpt["val_loss"],
        "total_training_time_s": round(total_training_time, 2),
        "best_checkpoint_path": best_ckpt_path,
        "latest_checkpoint_path": latest_ckpt_path,
        "splits": split_results,
        "training_history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Transformer Feature Extractor (Option B)")
    parser.add_argument("--highway", choices=["highway_2", "highway_7", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    start_total = time.time()
    print("=================================================================")
    print("Phase 2: Training Transformer Feature Extractor & Embedding Extraction")
    print("=================================================================")

    target_highways = HIGHWAYS if args.highway == "all" else [args.highway]
    summary = {
        "phase": "Phase 2 — Transformer Feature Extraction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
            "patience": args.patience,
            "seed": args.seed,
        },
        "highways": {},
    }

    for hw in target_highways:
        res = train_highway(
            highway=hw,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            resume=args.resume,
            seed=args.seed,
        )
        summary["highways"][hw] = res

    os.makedirs(EMB_BASE, exist_ok=True)
    summary_path = os.path.join(EMB_BASE, "embedding_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    elapsed_total = time.time() - start_total
    print(f"\n=================================================================")
    print(f"Phase 2 Execution Complete in {elapsed_total:.2f} s.")
    print(f"Summary written to: {summary_path}")
    print(f"=================================================================")


if __name__ == "__main__":
    main()
