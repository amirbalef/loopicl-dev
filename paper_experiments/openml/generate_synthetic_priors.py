"""Generate and save synthetic prior datasets using the local TabICL v2 prior.

Each dataset is saved as a pickle file in the output directory.
Run this once before benchmarking on synthetic data.

Prior types
-----------
    mlp_scm     MLP-based causal SCM
    tree_scm    Tree-based causal SCM
    mix_scm     Probabilistic mix of mlp_scm / tree_scm (default)
    graph_scm   DAG-based graph SCM (uses PriorConfig options below)

Graph SCM options (only relevant for graph_scm)
    --graph-noise               Add Gaussian noise to DAG nodes
    --allow-act-warping         Include random activations as num-feature warping
    --min-n-nodes INT           Min nodes in the causal graph (default: 2)
    --max-n-nodes INT           Max nodes in the causal graph (default: 32)

Filtering options
    --filter-unpredictable-datasets   Drop datasets where ExtraTrees <= dummy model
    --filter-unpredictable-graphs     Drop graphs where y is independent of x

Usage
-----
    # graph_scm with filtering
    uv run benchmark_eval/openml/generate_synthetic_priors.py \\
        --n-datasets 32 --prior-type graph_scm \\
        --filter-unpredictable-datasets --filter-unpredictable-graphs \\
        --output results/synthetic_priors/graph_scm

Output
------
    <output>/
        {idx:06d}.pkl   # one per synthetic dataset
        metadata.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Make the local src/ importable so we use ticla.priors.tabicl_v2 (not installed tabicl)
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))


def build_prior(args, output_dir: Path):
    from loopicl.priors.tabicl_v2 import PriorDataset
    from loopicl.priors.tabicl_v2.graph_lib._config import PriorConfig

    config = PriorConfig(
        add_gaussian_noise=args.graph_noise,
        allow_act_warping=args.allow_act_warping,
        min_n_nodes=args.min_n_nodes,
        max_n_nodes=args.max_n_nodes,
        filter_unpredictable_datasets=args.filter_unpredictable_datasets,
        filter_unpredictable_graphs=args.filter_unpredictable_graphs,
    )

    metadata = {
        "n_datasets": args.n_datasets,
        "prior_type": args.prior_type,
        "max_features": args.max_features,
        "max_classes": args.max_classes,
        "min_seq_len": args.min_seq_len,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "source": "ticla.priors.tabicl_v2",
        "graph_noise": args.graph_noise,
        "allow_act_warping": args.allow_act_warping,
        "min_n_nodes": args.min_n_nodes,
        "max_n_nodes": args.max_n_nodes,
        "filter_unpredictable_datasets": args.filter_unpredictable_datasets,
        "filter_unpredictable_graphs": args.filter_unpredictable_graphs,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    prior = PriorDataset(
        batch_size=args.batch_size,
        batch_size_per_gp=4,
        min_features=2,
        max_features=args.max_features,
        max_classes=args.max_classes,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        log_seq_len=True,
        seq_len_per_gp=False,
        min_train_size=0.3,
        max_train_size=0.9,
        prior_type=args.prior_type,
        config=config,
        n_jobs=1,
        device="cpu",
    )
    return prior


def generate_and_save(args) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    prior = build_prior(args, output_dir)

    saved = 0
    pbar = tqdm(total=args.n_datasets, desc="Generating synthetic datasets")

    # Resume: count already-existing dataset files
    if not args.overwrite:
        existing = sorted(output_dir.glob("[0-9]" * 6 + ".pkl"))
        saved = len(existing)
        if saved > 0:
            print(f"Resuming from index {saved} ({saved} datasets already exist).")
            pbar.update(saved)

    while saved < args.n_datasets:
        X, y, d, seq_lens, train_sizes = prior.get_batch()
        X = X.cpu().numpy()            # (batch_size, seq_len, max_features)
        y = y.cpu().numpy()            # (batch_size, seq_len)
        d = d.cpu().numpy()            # (batch_size,)
        seq_lens = seq_lens.cpu().numpy()
        train_sizes = train_sizes.cpu().numpy()

        for i in range(len(d)):
            if saved >= args.n_datasets:
                break

            out_file = output_dir / f"{saved:06d}.pkl"
            if out_file.exists() and not args.overwrite:
                saved += 1
                pbar.update(1)
                continue

            seq_len = int(seq_lens[i])
            n_feat = int(d[i])
            train_size = int(train_sizes[i])

            X_i = X[i, :seq_len, :n_feat]   # (seq_len, n_feat)
            y_i = y[i, :seq_len]             # (seq_len,)

            n_classes = int(len(np.unique(y_i)))
            problem_type = "binary" if n_classes == 2 else "multiclass"

            data = {
                "dataset_id":   saved,
                "X":            X_i.astype(np.float32),
                "y":            y_i.astype(np.int64),
                "train_size":   train_size,
                "n_features":   n_feat,
                "n_classes":    n_classes,
                "problem_type": problem_type,
                "seq_len":      seq_len,
            }

            with open(out_file, "wb") as f:
                pickle.dump(data, f)

            saved += 1
            pbar.update(1)

    pbar.close()
    print(f"\nSaved {saved} synthetic datasets to {output_dir}")
    print(
        f"Run benchmark:  uv run benchmark_eval/openml/run_benchmark.py "
        f"--dataset synthetic-priors --prior-dir {output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core options
    parser.add_argument("--n-datasets", type=int, default=32,
                        help="Number of synthetic datasets to generate (default: 32)")
    parser.add_argument("--output", default="results/synthetic_priors/tabicl_v2",
                        help="Output directory (default: results/synthetic_priors/tabicl_v2)")
    parser.add_argument("--prior-type", default="graph_scm",
                        choices=["mlp_scm", "tree_scm", "mix_scm", "graph_scm"],
                        help="Prior type (default: graph_scm)")
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--min-seq-len", type=int, default=100,
                        help="Minimum dataset size (default: 100)")
    parser.add_argument("--max-seq-len", type=int, default=1024,
                        help="Maximum dataset size (default: 1024)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Generation batch size — datasets per prior.get_batch() call (default: 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate all files even if they already exist")

    # --- Graph SCM ---
    graph = parser.add_argument_group("Graph SCM options (graph_scm prior type)")
    graph.add_argument("--graph-noise", action="store_true",
                       help="Add Gaussian noise to DAG nodes")
    graph.add_argument("--allow-act-warping", action="store_true",
                       help="Include random activations as a numerical feature warping method")
    graph.add_argument("--min-n-nodes", type=int, default=2,
                       help="Minimum number of nodes in the causal graph (default: 2)")
    graph.add_argument("--max-n-nodes", type=int, default=32,
                       help="Maximum number of nodes in the causal graph (default: 32)")

    # --- Filtering ---
    filt = parser.add_argument_group("Filtering options")
    filt.add_argument("--filter-unpredictable-datasets", action="store_true",
                      help="Remove datasets where ExtraTrees cannot beat a dummy model")
    filt.add_argument("--filter-unpredictable-graphs", action="store_true",
                      help="Filter graphs where y is independent of x")

    return parser.parse_args()


if __name__ == "__main__":
    generate_and_save(parse_args())
