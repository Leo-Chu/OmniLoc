# UniLoc — Training `main.py`

This project trains a **LocalizationTransformer** (PyTorch Lightning) for indoor localization from Wi‑Fi features (CSI, RSSI, SNR, AP vectors). Training can run as a **single phase** or as a **multi-stage** schedule defined in the YAML config.

## Prerequisites

- **Python** with PyTorch, **PyTorch Lightning**, **PyYAML**, and **NumPy** installed.
- **GPU** recommended; set `system.accelerator` and `system.devices` in the config (`cpu` works but is slower).

## Data layout

Before training, place pickle datasets under the directory given by `data.data_dir` (default: `data/` next to `main.py`):

| File | Role |
|------|------|
| `training.pkl` | Training split |
| `eval.pkl` | Validation |
| `test.pkl` | Held-out test (also used when you run evaluation) |

Paths are resolved relative to the repository root unless you use absolute paths.

## Configuration

Default config: **`cfg/localization_config.yaml`**.

```bash
python main.py --config path/to/your_config.yaml
```

Multi-stage behavior lives under the **`two_stage`** section:

- **`enabled: true`** — run the staged pipeline (stage 1 → stage 2, and stage 3 if enabled).
- **`use_three_stage: true`** — after stage 2, run stage 3 (requires a `stage3` block in the same YAML).

You can override the YAML with flags:

- **`--single-stage`** — ignore `two_stage.enabled` and run **one** training run using the top-level `training.*` settings (checkpointing on regression loss as in code).
- **`--two-stage`** — force multi-stage training even if `enabled` is false (still needs valid `stage1` / `stage2` blocks; use `stage3` when using three stages).

---

## Three-stage training (recommended schedule)

When **`two_stage.enabled`** and **`two_stage.use_three_stage`** are both **true**, `main.py` runs **three** phases in order. Each stage has its own hyperparameters under `two_stage.stage1`, `two_stage.stage2`, and `two_stage.stage3` in the YAML.

### Stage 1 — Classification-oriented pretraining (`s1_cls`)

- **Goal:** Train building/floor classification together with a **smaller** regression weight so representation and class heads learn first (multi-task, V2Ab-like coupling when `detach_logits_for_regression: false`).
- **Checkpointing:** Monitors **`val/mean_cls_acc`** (maximize) by default — see `stage1.monitor` in YAML.
- **Output directory:** `results/logs/<experiment_name>/checkpoints/stage1_cls/`
- **Stable alias for the next stage:** `stage1_pretrained.ckpt` (copy of the best stage‑1 checkpoint).

### Stage 2 — Regression focus (`s2_reg`)

- **Goal:** Emphasize **coordinate regression**; classification CE terms are typically **turned off** (`building_weight` / `floor_weight` to `0` in the sample config). **`freeze_classifier_heads: true`** freezes building/floor heads so only the backbone and regression path update.
- **Loads:** Best **stage 1** weights from `stage1_pretrained.ckpt` or the latest checkpoint under `stage1_cls/`.
- **Checkpointing:** **`val/reg_loss`** (minimize).
- **Output directory:** `stage2_reg/`
- **Stable alias:** `stage2_pretrained.ckpt`

### Stage 3 — Joint fine-tuning (`s3_ft`)

- **Goal:** **Joint** fine-tune with regression + **light** classification (`building_weight` / `floor_weight` small but non-zero in the sample config). Heads are typically **unfrozen** (`freeze_classifier_heads: false`) for calibration and joint optimization.
- **Loads:** Best **stage 2** checkpoint from `stage2_pretrained.ckpt` or `stage2_reg/`.
- **Checkpointing:** **`val/reg_loss`** (minimize).
- **Output directory:** `stage3_ft/`

After all stages, the run saves evaluation artifacts and **`best_eval_results.txt`** under the experiment log directory (see `logging.log_dir`).

---

## Commands

### Full three-stage run (default when YAML has `enabled` + `use_three_stage`)

```bash
python main.py
```

Uses `cfg/localization_config.yaml` unless you pass `--config`.

### Run one stage only (resume / debug)

Stages expect prior checkpoints when not run from the full pipeline:

```bash
python main.py stage 1    # Classification pretrain only
python main.py stage 2    # Requires stage 1 (see stable paths above)
python main.py stage 3    # Requires stage 2
```

Aliases like `stage1` / `stage2` / `stage3` (no space) are also accepted by the parser.

**Do not** combine `stage 1` / `2` / `3` with **`--single-stage`**.

### Single-phase training (no staged schedule)

```bash
python main.py --single-stage
```

### Evaluation only (no training)

```bash
python main.py --test-only
```

Optionally pass **`--ckpt path/to/model.ckpt`** or **`--weights path/to/best_model.pt`**. If omitted, the script searches under the checkpoint root for a Lightning `.ckpt`, or falls back to `results/logs/best_model.pt` when present.

---

## Where outputs go

- **Logs & checkpoints:** `logging.log_dir` (default `./results/logs`), then `<experiment_name>/checkpoints/` with subfolders `stage1_cls`, `stage2_reg`, `stage3_ft` as applicable.
- **TensorBoard:** Under the same log root; run names include stage tags (e.g. `*_stage1_cls`, `*_stage2_reg`, `*_stage3_ft`).

---

## Troubleshooting

- **`FileNotFoundError` for `training.pkl` / `eval.pkl` / `test.pkl`** — create `data/` (or set `data.data_dir`) and add the three pickle files.
- **`No stage-1 checkpoint`** when running **stage 2** — run **`python main.py stage 1`** first, or run the full multi-stage command once.
- **`stage 3` requires `two_stage.stage3` in YAML** — add a `stage3` block or disable `use_three_stage`.

For exact loss weights, epochs, patience, and monitors, edit **`cfg/localization_config.yaml`** — the comments there mirror what `main.py` applies per stage.

---

## Other config files

- **`cfg/config.yaml`** — Example config for a generic encoder–decoder-style transformer experiment. It is **not** wired into `main.py`. Use **`cfg/localization_config.yaml`** for this project.

## Citations

If you use this code, the **WiLoc** dataset, or the **UniLoc** model in research, please cite the relevant work(s) below.

**WiLoc** (dataset):

```bibtex
@article{zhang2026wiloc,
  title={WiLoc: Massive Measured Dataset of Wi-Fi Channel State Information with Application to Machine-Learning Based Localization},
  author={Zhang, Yuning and Chu, Lei and Serbetci, Omer Gokalp and Gomez-Ponce, Jorge and Molisch, Andreas F},
  journal={arXiv preprint arXiv:2602.09115, IEEE International Conference on Computer Communications (INFOCOM)},
  year={2026}
}
```

**UniLoc** (foundation model for anchor-free UE localization):

```bibtex
@article{chu2026uniloc,
  title={UniLoc: A Geometry-Aware Foundation Model for Anchor-Free UE Localization Across Diverse Indoor Environments},
  author={Chu, Lei and Zhang, Yuning and Serbetci, Omer Gokalp and Bassel Abou Ali, Modad and Molisch, Andreas F},
  journal={under review},
  year={2026}
}
```

The second entry uses the key `chu2026uniloc` so it does not duplicate `zhang2026wiloc` in your `.bib` file; you may rename the key to match your bibliography style.

## License

This repository includes an **MIT** license in `LICENSE`. Update the copyright line if your institution or paper requires a different license or attribution.
