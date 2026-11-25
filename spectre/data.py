"""Explicit CSV schema, content validation and reproducible subject splits."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold

TASKS = [
    "RIGHT UPPER ARM SCORE", "LEFT UPPER ARM SCORE",
    "RIGHT LOWER ARM SCORE", "LEFT LOWER ARM SCORE",
    "RIGHT WRIST SCORE", "LEFT WRIST SCORE", "NECK SCORE", "TRUNK SCORE",
]
DIMS = [6, 6, 3, 3, 6, 6, 6, 6]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def image_payload(path):
    """Hash exactly the bytes decoded; pixel hash also detects metadata-only copies."""
    raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        rgb.load()
    pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    return rgb, hashlib.sha256(raw).hexdigest(), pixel_hash


def load_table(csv_path, label_base, subject_col="user", image_root=None,
               image_extension=".jpg", extra_group_cols=()):
    """Require explicit labels and identity. Reject duplicates instead of silently dropping."""
    if label_base not in (0, 1):
        raise ValueError("label_base must explicitly be 0 or 1")
    if not image_extension.startswith(".") or "/" in image_extension or "\\" in image_extension:
        raise ValueError("image_extension must be a suffix such as .jpg")
    path = Path(csv_path).expanduser().resolve(strict=True)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names")
    groups = list(dict.fromkeys([subject_col, *extra_group_cols]))
    if any(g in ["ID", "_path", "_sha256", "_pixel_sha256", "fold", *TASKS] for g in groups):
        raise ValueError("Group columns must be distinct from ID and targets")
    required = ["ID", *TASKS, *groups]
    if not set(required).issubset(header):
        raise ValueError(f"Missing CSV columns: {sorted(set(required) - set(header))}")
    df = pd.read_csv(path, dtype={c: "string" for c in ["ID", *groups]})
    if df.empty:
        raise ValueError("Empty dataset")
    df = df[required].copy()
    for column in ["ID", *groups]:
        if df[column].isna().any() or df[column].str.strip().eq("").any():
            raise ValueError(f"Missing/blank values in {column}")
        if df[column].ne(df[column].str.strip()).any():
            raise ValueError(f"Leading/trailing whitespace in {column}; correct the CSV explicitly")
    if df.ID.duplicated().any():
        raise ValueError("Duplicate image IDs")
    for task, count in zip(TASKS, DIMS):
        values = pd.to_numeric(df[task], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Targets must be finite integers: {task}")
        if ((values < label_base) | (values >= label_base + count)).any():
            raise ValueError(f"Out-of-range target for {task}, expected {label_base}..{label_base+count-1}")
        df[task] = (values - label_base).astype(np.int64)
    base = Path(image_root).expanduser().resolve(strict=True) if image_root else path.parent
    resolved, raw_hashes, pixel_hashes = [], [], []
    for identifier in df.ID:
        image_path = Path(identifier).expanduser()
        if not image_path.suffix:
            image_path = image_path.with_suffix(image_extension)
        if not image_path.is_absolute():
            image_path = base / image_path
        image_path = image_path.resolve(strict=True)
        if not image_path.is_file():
            raise ValueError(f"Not an image file: {image_path}")
        _, raw_hash, pixel_hash = image_payload(image_path)
        resolved.append(str(image_path)); raw_hashes.append(raw_hash); pixel_hashes.append(pixel_hash)
    if len(set(resolved)) != len(resolved) or len(set(pixel_hashes)) != len(pixel_hashes):
        raise ValueError("Duplicate image paths or identical decoded RGB images; curate duplicates before splitting")
    df["_path"] = resolved
    df["_sha256"] = raw_hashes
    df["_pixel_sha256"] = pixel_hashes
    return df.sort_values("ID", kind="stable").reset_index(drop=True)


def make_splits(df, protocol, subject_col, seed, val_fraction=0.2, extra_group_cols=()):
    """Outer test is subject-disjoint; inner holdout is chosen without target labels."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must lie strictly between 0 and 1")
    users = sorted(df[subject_col].unique())
    minimum = 5 if protocol == "group5" else 3
    if protocol not in ("loso", "group5") or len(users) < minimum:
        raise ValueError(f"{protocol} requires at least {minimum} distinct subjects")
    if protocol == "loso":
        outer = [(np.flatnonzero(df[subject_col].ne(u)), np.flatnonzero(df[subject_col].eq(u))) for u in users]
    else:
        outer = list(GroupKFold(n_splits=5).split(df, groups=df[subject_col]))
    folds = []
    for number, (dev_idx, test_idx) in enumerate(outer, 1):
        development = sorted(df.iloc[dev_idx][subject_col].unique())
        rng = np.random.default_rng(seed + number)
        n_val = min(len(development)-1, max(1, int(np.ceil(val_fraction*len(development)))))
        val_users = set(rng.permutation(development)[:n_val])
        val_idx = np.array([i for i in dev_idx if df.iloc[i][subject_col] in val_users], dtype=int)
        train_idx = np.array([i for i in dev_idx if df.iloc[i][subject_col] not in val_users], dtype=int)
        split = {"train": train_idx.tolist(), "validation": val_idx.tolist(), "test": test_idx.tolist()}
        for name, indices in split.items():
            if not indices:
                raise ValueError(f"Fold {number}: empty {name}")
        for col in ["ID", subject_col, *extra_group_cols]:
            values = [set(df.iloc[split[name]][col]) for name in split]
            if any(values[i] & values[j] for i in range(3) for j in range(i+1, 3)):
                raise ValueError(f"Fold {number}: shared {col} between partitions. Group jointly or correct identifiers.")
        folds.append({"fold": number, "indices": split,
                      "subjects": {name: sorted(df.iloc[idx][subject_col].unique()) for name, idx in split.items()}})
    all_test = [i for fold in folds for i in fold["indices"]["test"]]
    if sorted(all_test) != list(range(len(df))):
        raise ValueError("Outer folds do not cover every sample exactly once")
    return folds


def records(df):
    """Portable identity: exclude resolved absolute paths but include image bytes and labels."""
    return df.drop(columns="_path").to_dict(orient="records")


class PostureDataset(Dataset):
    def __init__(self, frame, image_size=224):
        self.frame = frame.reset_index(drop=True).copy()
        self.image_size = image_size

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image, raw_hash, _ = image_payload(row["_path"])
        if raw_hash != row["_sha256"]:
            raise ValueError(f"Image changed after validation: {row['ID']}")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.0
        labels = torch.tensor(row[TASKS].to_numpy(dtype=np.int64), dtype=torch.long)
        return tensor, labels
