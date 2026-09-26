"""Entry point for generating and saving prior datasets.

Usage:
    python -m loopicl.priors.tabicl_v2.genload --model_config loopicl.configs.nanoticla_priors_v2
    python -m loopicl.priors.tabicl_v2.genload --model_config loopicl.configs.nanoticla_priors_v2 --save_dir /override/path

--model_config accepts a dotted Python module path that contains a PriorConfig dataclass.
Command-line arguments always override values from the config module.

Field mapping from PriorConfig:
  prior_device -> device
All other PriorConfig fields that match genload argument names are applied directly.
Fields not recognised by the parser (e.g. delete_after_load, prior_dir) are ignored.
"""

import dataclasses
import importlib
import argparse

from ._genload import SavePriorDataset
from .graph_lib._config import PriorConfig


def str2bool(value):
    return value.lower() == "true"


def train_size_type(value):
    """Custom type function to handle both int and float train sizes."""
    value = float(value)
    if 0 < value < 1:
        return value
    elif value.is_integer():
        return int(value)
    else:
        raise argparse.ArgumentTypeError(
            "Train size must be either an integer (absolute position) "
            "or a float between 0 and 1 (ratio of sequence length)."
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Generate training prior datasets")
    parser.add_argument("--model_config", type=str, default=None,
                        help="Dotted Python module path containing a PriorConfig dataclass "
                             "(e.g. loopicl.configs.nanoticla_priors_v2)")
    parser.add_argument("--save_dir", type=str, default="data", help="Directory to save the generated data")
    parser.add_argument("--np_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument("--torch_seed", type=int, default=42, help="Random seed for torch")
    parser.add_argument("--num_batches", type=int, default=10000, help="Number of batches to generate")
    parser.add_argument("--resume_from", type=int, default=0, help="Resume generation from this batch index")
    parser.add_argument("--regression", default=False, type=str2bool, help="If True, generate regression datasets")
    parser.add_argument("--batch_size", type=int, default=512, help="Total batch size")
    parser.add_argument("--batch_size_per_gp", type=int, default=4, help="Batch size per group")
    parser.add_argument("--min_features", type=int, default=2, help="Minimum number of features")
    parser.add_argument("--max_features", type=int, default=100, help="Maximum number of features")
    parser.add_argument("--max_classes", type=int, default=10, help="Maximum number of classes")
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum sequence length")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument(
        "--log_seq_len", default=False, type=str2bool,
        help="If True, sample sequence length from log-uniform distribution between min_seq_len and max_seq_len",
    )
    parser.add_argument(
        "--log_n_features", default=False, type=str2bool,
        help="If True, sample number of features from log-uniform distribution between max_features and min_features",
    )
    parser.add_argument(
        "--seq_len_per_gp", default=False, type=str2bool,
        help="If True, sample sequence length independently for each group",
    )
    parser.add_argument(
        "--min_train_size", type=train_size_type, default=0.1, help="Minimum training size position/ratio"
    )
    parser.add_argument(
        "--max_train_size", type=train_size_type, default=0.9, help="Maximum training size position/ratio"
    )
    parser.add_argument(
        "--replay_small", default=False, type=str2bool,
        help="If True, occasionally sample smaller sequence lengths to ensure model robustness on smaller datasets",
    )
    parser.add_argument(
        "--prior_type", type=str, default="graph_scm",
        choices=["mlp_scm", "tree_scm", "mix_scm", "graph_scm"],
        help="Type of prior to use",
    )
    PriorConfig.add_args_to_parser(parser)
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs for parallel processing")
    parser.add_argument("--num_threads_per_generate", type=int, default=1, help="Threads per generation")
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for generation"
    )
    return parser


def _load_prior_config_defaults(module_path: str) -> dict:
    """Import a config module and extract PriorConfig fields as a dict.

    The PriorConfig dataclass field `prior_device` is renamed to `device`.
    Fields not recognised by the genload parser are silently ignored later.

    If the module cannot be found as-is, it is retried with the
    ``loopicl.configs.`` prefix automatically.
    """
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        mod = importlib.import_module(f"loopicl.configs.{module_path}")
    if not hasattr(mod, "PriorConfig"):
        raise AttributeError(f"Module '{module_path}' has no PriorConfig dataclass")

    cfg = mod.PriorConfig()
    cfg_dict = dataclasses.asdict(cfg)

    # Rename prior_device -> device if present
    if "prior_device" in cfg_dict:
        cfg_dict["device"] = cfg_dict.pop("prior_device")

    return cfg_dict


def main():
    # Pre-parse only --model_config so we can load it before full parsing
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--model_config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = build_parser()

    if pre_args.model_config is not None:
        cfg_dict = _load_prior_config_defaults(pre_args.model_config)

        # Only pass keys that are valid parser arguments
        valid_keys = {a.dest for a in parser._actions}
        filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        parser.set_defaults(**filtered)

    args = parser.parse_args()
    saver = SavePriorDataset(args)
    saver.run()


if __name__ == "__main__":
    main()
