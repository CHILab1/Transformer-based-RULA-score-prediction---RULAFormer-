# RULAFormer — Transformer-based RULA Score Prediction

A multi-task Vision Transformer implementation for predicting eight body-segment postural scores from images. The model uses a SimpleViT encoder and eight task-specific gated classification heads.

The outputs are the eight component scores listed below. This implementation does not compute a final combined RULA score from those outputs.

## Repository layout

```text
.
├── spectre/
│   ├── cli.py                   # Experiment orchestration and checkpoint verification
│   ├── data.py                  # CSV validation, images and subject-disjoint splits
│   ├── engine.py                # Training, internal validation and prediction
│   ├── metrics.py               # Per-class, per-task and cross-fold reporting
│   └── model.py                 # Multi-task SimpleViT
├── 5fold_cross_validation/
│   └── main_trasformer_5fold.py  # Entry point for grouped 5-fold
├── loso_cross_validation/
│   └── main_trasformer_loso.py   # Entry point for LOSO
├── tests/
│   └── test_protocol.py
├── requirements.txt
```


## CSV contract

The CSV must contain `ID`, an explicit subject identity column (default `user`), and these **named** targets in any column order:

| Target | Observed score range |
|---|---|
| RIGHT UPPER ARM SCORE | 1–5 |
| LEFT UPPER ARM SCORE | 1–5 |
| RIGHT LOWER ARM SCORE | 1–3 |
| LEFT LOWER ARM SCORE | 1–3 |
| RIGHT WRIST SCORE | 2–6 |
| LEFT WRIST SCORE | 2–6 |
| NECK SCORE | 1–6 |
| TRUNK SCORE | 1–6 |

`--label_base 0` means input labels are 0…C−1. `--label_base 1` means they are 1…C. This argument is mandatory; no heuristic guesses the label encoding. All targets must be finite integers in their declared range. Internally targets are zero-based; predictions and confusion matrices are exported as scores 1…C, independently of the CSV encoding. These cardinalities reproduce the supplied architecture; verify them against the dataset's annotation specification before use.

`user` must identify the same physical person across all sessions, cameras and aliases. It is never inferred from directories. Missing identities, targets, duplicate column names, duplicated paths and identical decoded RGB images are rejected. Unused CSV columns are ignored; targets are selected by name. ID values and group identities must have no leading/trailing whitespace.

`ID` may be an absolute image path or a path relative to `--image_root` (default: CSV directory). A missing suffix is filled with `--image_extension` (default `.jpg`); existing suffixes are retained. Actual image files must exist and decode. Images are EXIF-oriented, converted to RGB, bilinearly resized to a square and scaled to [0,1]. This explicitly changes the original BGR/tensor-resize preprocessing, so legacy checkpoints cannot be reused.

Image bytes and decoded pixels are hashed during validation. Changed image bytes are rejected when loaded during training/inference. Hashes do not identify recompressed near-duplicates or incorrectly assigned identities: audit acquisitions and label provenance separately. Exact duplicates are rejected for explicit dataset curation, never silently removed.

If multiple subjects share an acquisition that must remain independent, provide its globally unique identifier with `--extra_group_cols acquisition_id`. The runner rejects any split crossing that group rather than silently accepting it. Local session names such as `session_1` must first be made globally unique. Connected acquisitions spanning subjects may require a separately designed grouping protocol; the runner does not pretend that subject-only splitting solves such a case.

## Protocols

```text
All subjects
    ├── Outer test subjects ──────────────> Final evaluation only
    └── Remaining subjects
            ├── Inner training subjects ─> Parameter updates
            └── Inner validation subjects > Checkpoint selection / early stopping
```


- `loso`: each subject is the outer test subject once; requires at least three subjects.
- `group5`: five outer GroupKFold partitions by subject; requires at least five subjects. This replaces the original random-frame 5-fold. GroupKFold is deterministic on the canonical input; `--seed` controls internal validation and training, not outer group allocation.

For each outer fold, a deterministic random subset of the remaining subjects is held out for internal validation. Its size is `ceil(val_fraction × remaining_subjects)`, clamped to leave at least one training and one validation subject. The default fraction is 0.2. Selection uses no label values. Subject names and image IDs are sorted canonically; reordering CSV rows does not change splits or checkpoint identities.

The model is initialized independently for each fold with seed `seed + fold_index`. It trains only on the inner training subjects. The checkpoint minimizing **internal validation loss** is selected, with earlier epochs winning exact ties. Early stopping monitors that same criterion. The internal validation subjects are not subsequently added to training. This is an outer cross-validation protocol with an inner holdout, **not a full inner cross-validation hyperparameter search**. It trades some training data for lower computational cost.

After checkpoint selection the outer test is evaluated once. No test loader is accepted by the training function. The same subject-disjoint rule applies to inner validation. Each image is outer-test exactly once for a full run. Running one fold produces an explicitly partial summary. Do not tune architectures, hyperparameters, seeds or preprocessing by inspecting outer-test results; use a separate development design or nested inner CV if model selection is required.

