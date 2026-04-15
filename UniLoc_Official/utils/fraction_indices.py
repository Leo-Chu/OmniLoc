"""
Fraction indices utility: save and reuse exact same subset of data across case studies.

IMPORTANT: Two different fraction semantics - do NOT conflate:

1. SUPERVISED_FRACTIONS: For supervised learning (train from scratch).
   - Fraction = percentage of LABELED DATA to use.
   - Base: full labeled training set (e.g. training_with_f1_in_b1.pkl).
   - 20% = use 20% of all labeled samples.

2. FINETUNE_FRACTIONS: For finetuning (adapt pretrained model).
   - Applied WITHIN a base (e.g. 60% supervised data, N_train=3488).
   - base_fraction: e.g. 0.6 = 60% of full data (matches supervised 60% case).
   - inner_fraction: [1%, 10%, 20%, 40%, 80%, 100%] of that base.
   - Example: 60% base (3488 samples) -> finetune on 20% of 3488 = 698 samples.
"""

import pickle
from pathlib import Path

import torch


DEFAULT_CACHE_DIR = Path("C:/coding/Localization/logs/fraction_indices_cache")

# Supervised: fraction of labeled data (train from scratch)
SUPERVISED_FRACTIONS = [0.20, 0.40, 0.60, 0.80]

# Finetuning: fraction of taken labeled data (adapt pretrained model)
FINETUNE_FRACTIONS = [0.01, 0.10, 0.20, 0.40, 0.80, 1.0]  # 1% = extreme low-data case


def get_fraction_indices(
    n_total: int,
    fraction: float,
    seed: int = 42,
    cache_dir: Path | str | None = None,
    dataset_key: str | None = None,
) -> list[int]:
    """
    Get indices for a fraction of data. Load from cache if available, else compute and save.

    Args:
        n_total: Total number of samples in the dataset.
        fraction: Fraction to use (0.0 to 1.0), e.g. 0.2 for 20%.
        seed: Random seed for reproducibility.
        cache_dir: Directory to save/load indices. If None, uses DEFAULT_CACHE_DIR.
        dataset_key: Identifier for the dataset. Use distinct keys for different
                     fraction semantics, e.g. "f1_in_b1_supervised" vs "f1_in_b1_finetune".
                     If None, cache is not used.

    Returns:
        List of indices to use (length = max(1, int(n_total * fraction))).
    """
    n_use = max(1, int(n_total * fraction))
    pct = int(fraction * 100)

    if cache_dir is not None and dataset_key is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        subdir = cache_dir / dataset_key
        subdir.mkdir(parents=True, exist_ok=True)
        cache_path = subdir / f"frac{pct}_n{n_total}_seed{seed}.pkl"

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                indices = pickle.load(f)
            if len(indices) == n_use:
                return indices
            # Dataset size changed; fall through to recompute

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()[:n_use]

    if cache_dir is not None and dataset_key is not None:
        with open(cache_path, "wb") as f:
            pickle.dump(indices, f)

    return indices


def get_fraction_indices_within_base(
    n_total: int,
    base_fraction: float,
    inner_fraction: float,
    seed: int = 42,
    cache_dir: Path | str | None = None,
    dataset_key: str | None = None,
) -> list[int]:
    """
    Get indices for inner_fraction of a base subset (base_fraction of n_total).
    Used when finetuning within a supervised base (e.g. 60% supervised -> finetune on
    [1%, 10%, 20%, 40%, 80%, 100%] of that 60%).

    Args:
        n_total: Total dataset size.
        base_fraction: Base subset fraction (e.g. 0.6 for 60% supervised case).
        inner_fraction: Fraction within base (e.g. 0.2 for 20% of base).
        seed, cache_dir, dataset_key: Same as get_fraction_indices.

    Returns:
        Indices into full dataset (length = inner_fraction of base size).
    """
    base_indices = get_fraction_indices(
        n_total, base_fraction, seed, cache_dir,
        f"{dataset_key}_base{int(base_fraction*100)}" if dataset_key else None,
    )
    n_base = len(base_indices)
    inner_indices = get_fraction_indices(
        n_base, inner_fraction, seed, cache_dir,
        f"{dataset_key}_b{int(base_fraction*100)}" if dataset_key else None,
    )
    return [base_indices[i] for i in inner_indices]
