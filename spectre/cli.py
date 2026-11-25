"""Command-line entrypoint. Existing runs are immutable and never silently resumed."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

import pandas as pd
import torch

from .data import TASKS, DIMS, PostureDataset, digest, load_table, make_splits, records
from .engine import fit, loader, predict, set_seed, write_json
from .metrics import export_metrics, export_fold_summary
from .model import SimpleViT


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def source_hash():
    return digest({p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def environment():
    return {"python": sys.version, "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ["torch", "numpy", "pandas", "scikit-learn", "Pillow"]},
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}


def device_from_name(name):
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This validated runner supports cpu or cuda[:index]")
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device is unavailable: {name}")
    return device


def parser(default_protocol=None):
    p = argparse.ArgumentParser(description="SPECTRE: outer subject-disjoint CV with inner subject holdout")
    p.add_argument("--protocol", choices=["loso", "group5"], default=default_protocol, required=default_protocol is None)
    p.add_argument("--csv_path", required=True)
    p.add_argument("--output_dir", required=True, help="New directory; existing directories are rejected")
    p.add_argument("--label_base", type=int, choices=[0, 1], required=True)
    p.add_argument("--subject_col", default="user", help="Explicit, globally unique person identifier")
    p.add_argument("--extra_group_cols", nargs="*", default=[], help="Additional global acquisition IDs that must not cross partitions")
    p.add_argument("--image_root", default=None, help="Base for relative ID paths; default: CSV directory")
    p.add_argument("--image_extension", default=".jpg", help="Appended only when ID has no suffix")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold", default="all", help="One-based fold index or all")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--val_fraction", type=float, default=0.2, help="Fraction of development subjects for internal validation")
    p.add_argument("--device", default="auto")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--weights_dir", help="Root of a completed run from this version; legacy weights are rejected")
    p.add_argument("--validate_only", action="store_true", help="Validate images/schema and export split manifest without training")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=32)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--dim_head", type=int, default=64)
    p.add_argument("--mlp_dim", type=int, default=2048)
    return p


def validate_arguments(args):
    for name in ["epochs", "batch_size", "patience", "image_size", "patch_size", "dim", "depth", "heads", "dim_head", "mlp_dim"]:
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.dim < 8 or args.dim % 4 or args.image_size % args.patch_size:
        raise ValueError("dim must be >=8 and divisible by 4; image_size must be divisible by patch_size")
    if not 0 <= args.seed < 2**32 - 1000000 or args.num_workers < 0:
        raise ValueError("Invalid seed or num_workers")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 < args.val_fraction < 1:
        raise ValueError("Invalid lr or val_fraction")
    if args.eval_only != (args.weights_dir is not None):
        raise ValueError("--eval_only and --weights_dir must be supplied together")
    if args.eval_only and args.validate_only:
        raise ValueError("Choose evaluation or validation-only")


def checkpoint_preflight(folder, expected):
    folder = Path(folder)
    weights, metadata = folder / "best.pt", folder / "checkpoint.json"
    if not weights.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"Missing checkpoint or provenance: {folder}")
    saved = json.loads(metadata.read_text())
    if saved.get("signature") != expected:
        raise ValueError(f"Checkpoint data/split/config/source mismatch: {folder}")
    if saved.get("weights_sha256") != sha256_file(weights):
        raise ValueError(f"Checkpoint bytes differ from recorded hash: {folder}")
    return saved


def main(argv=None, default_protocol=None):
    args = parser(default_protocol).parse_args(argv)
    validate_arguments(args)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    # CUDA determinism configuration must precede device discovery/context creation.
    set_seed(args.seed)
    device = device_from_name(args.device)
    df = load_table(args.csv_path, args.label_base, args.subject_col, args.image_root,
                    args.image_extension, args.extra_group_cols)
    folds = make_splits(df, args.protocol, args.subject_col, args.seed, args.val_fraction, args.extra_group_cols)
    selected = folds if args.fold == "all" else [f for f in folds if str(f["fold"]) == args.fold]
    if not selected:
        raise ValueError(f"fold must be all or an integer from 1 to {len(folds)}")
    model_config = {name: getattr(args, name) for name in ["image_size", "patch_size", "dim", "depth", "heads", "dim_head", "mlp_dim"]}
    model_config.update(num_classes=len(TASKS), output_class_dims=DIMS)
    config = {name: getattr(args, name) for name in ["protocol", "label_base", "subject_col", "extra_group_cols", "seed", "epochs", "batch_size", "lr", "patience", "val_fraction"]}
    config.update(model=model_config, optimizer="SGD_no_momentum", selection="internal_validation_loss",
                  preprocessing="Pillow_EXIF_RGB_bilinear_square_float32_0_1", source_sha256=source_hash(),
                  dataset_sha256=digest(records(df)), format_version=2)
    manifests = {}
    for fold in selected:
        contents = {name: records(df.iloc[idx]) for name, idx in fold["indices"].items()}
        manifests[fold["fold"]] = {"config": config, "fold": fold["fold"], "subjects": fold["subjects"],
                                    "partitions": contents, "seed": args.seed + fold["fold"]}
    signatures = {idx: digest(manifest) for idx, manifest in manifests.items()}
    # Verify every requested checkpoint before inference, not after some fold results are written.
    if args.eval_only:
        for fold in selected:
            checkpoint_preflight(Path(args.weights_dir) / f"fold_{fold['fold']}", signatures[fold["fold"]])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", {"config": config, "environment": environment(), "arguments": vars(args),
                                     "selected_folds": [f["fold"] for f in selected], "total_folds": len(folds),
                                     "status": "validated" if args.validate_only else "running"})
    for idx, manifest in manifests.items():
        write_json(output / f"manifest_fold_{idx}.json", manifest)
    if args.validate_only:
        print(f"Validated {len(df)} images, {len(folds)} outer folds. Manifests: {output}")
        return
    metric_rows, predictions = [], []
    for fold in selected:
        idx, split = fold["fold"], fold["indices"]
        seed = args.seed + idx
        set_seed(seed)
        model = SimpleViT(**model_config).to(device)
        folder = output / f"fold_{idx}"
        print(f"Fold {idx}: train={fold['subjects']['train']}, inner validation={fold['subjects']['validation']}, outer test={fold['subjects']['test']}", flush=True)
        if args.eval_only:
            source = Path(args.weights_dir) / f"fold_{idx}"
            metadata = checkpoint_preflight(source, signatures[idx])
            model.load_state_dict(torch.load(source / "best.pt", map_location=device, weights_only=True))
            folder.mkdir()
            write_json(folder / "evaluation_source.json", {"source": str(source.resolve()), **metadata})
        else:
            train_batches = loader(PostureDataset(df.iloc[split["train"]], args.image_size), args.batch_size, args.num_workers, seed, True)
            val_batches = loader(PostureDataset(df.iloc[split["validation"]], args.image_size), args.batch_size, args.num_workers, seed)
            selection = fit(model, train_batches, val_batches, device, args.epochs, args.patience, args.lr, folder)
            write_json(folder / "checkpoint.json", {"signature": signatures[idx], "selection": selection,
                                                     "weights_sha256": sha256_file(folder / "best.pt")})
        test_frame = df.iloc[split["test"]]
        test_batches = loader(PostureDataset(test_frame, args.image_size), args.batch_size, args.num_workers, seed)
        true, pred = predict(model, test_batches, device)
        if len(true) != len(test_frame):
            raise ValueError("Lost test samples")
        rows = export_metrics(true, pred, folder / "metrics")
        metric_rows.extend([{**row, "fold": idx} for row in rows])
        prediction = test_frame[["ID", args.subject_col]].reset_index(drop=True).copy()
        prediction["fold"] = idx
        for i, task in enumerate(TASKS):
            prediction[f"true::{task}"] = true[:, i] + 1
            prediction[f"pred::{task}"] = pred[:, i] + 1
        prediction.to_csv(folder / "predictions.csv", index=False)
        predictions.append(prediction)
        del model, test_batches
        if not args.eval_only:
            del train_batches, val_batches
    complete = len(selected) == len(folds)
    all_predictions = pd.concat(predictions, ignore_index=True)
    if all_predictions.ID.duplicated().any() or (complete and set(all_predictions.ID) != set(df.ID)):
        raise ValueError("Invalid out-of-fold prediction coverage")
    all_predictions.to_csv(output / "predictions.csv", index=False)
    export_fold_summary(metric_rows, output, complete)
    if complete:
        truth = all_predictions[[f"true::{task}" for task in TASKS]].to_numpy(dtype=int) - 1
        predicted = all_predictions[[f"pred::{task}" for task in TASKS]].to_numpy(dtype=int) - 1
        export_metrics(truth, predicted, output / "pooled_sample_metrics")
    write_json(output / "completion.json", {"status": "complete", "complete_protocol": complete,
                                            "evaluated_folds": len(selected), "expected_folds": len(folds),
                                            "test_samples": len(all_predictions)})
    print(f"Completed. Results: {output}")


if __name__ == "__main__":
    main()
