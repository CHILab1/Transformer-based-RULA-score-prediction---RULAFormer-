"""Synthetic regression tests; no research data or pretrained weights are required."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import TensorDataset

from spectre.cli import main, checkpoint_preflight
from spectre.data import TASKS, DIMS, PostureDataset, load_table, make_splits, records, digest
from spectre.engine import epoch_loss, fit, loader, predict, set_seed
from spectre.metrics import export_metrics, export_fold_summary
from spectre.model import SimpleViT, Patchify


def synthetic(folder, subjects=6, frames=3):
    rng = np.random.default_rng(13)
    rows = []
    for subject in range(subjects):
        for frame in range(frames):
            path = folder / f"s{subject}_f{frame}.png"
            Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(path)
            rows.append({"ID": path.name, "user": f"S{subject}", **{task: frame % count for task, count in zip(TASKS, DIMS)}})
    csv = folder / "data.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def tiny_model():
    return SimpleViT(image_size=8, patch_size=4, num_classes=8, dim=8, depth=1,
                     heads=1, dim_head=8, mlp_dim=16, output_class_dims=DIMS)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.csv = synthetic(self.folder)

    def tearDown(self):
        self.temp.cleanup()

    def test_splits_disjoint_complete_stable_and_label_independent(self):
        df = load_table(self.csv, 0)
        for protocol in ["loso", "group5"]:
            folds = make_splits(df, protocol, "user", 42)
            test_ids = []
            for fold in folds:
                groups = [set(df.iloc[indices].user) for indices in fold["indices"].values()]
                for i in range(3):
                    for j in range(i+1, 3):
                        self.assertFalse(groups[i] & groups[j])
                test_ids.extend(df.iloc[fold["indices"]["test"]].ID)
            self.assertCountEqual(test_ids, df.ID)
            changed = df.copy()
            changed[TASKS] = 0
            self.assertEqual(folds, make_splits(changed, protocol, "user", 42))
        raw = pd.read_csv(self.csv).iloc[::-1]
        raw.to_csv(self.csv, index=False)
        reordered = load_table(self.csv, 0)
        self.assertEqual(records(df), records(reordered))
        self.assertEqual(make_splits(df, "loso", "user", 42), make_splits(reordered, "loso", "user", 42))

    def test_target_schema_and_base(self):
        original = pd.read_csv(self.csv)
        expected = load_table(self.csv, 0)
        reordered = original[["user", *reversed(TASKS), "ID"]].copy()
        reordered[TASKS] += 1
        reordered["unused_metadata"] = "ignored"
        reordered.to_csv(self.csv, index=False)
        observed = load_table(self.csv, 1)
        np.testing.assert_array_equal(expected[TASKS], observed[TASKS])
        for value in [0.5, -1, 6, float("nan")]:
            bad = original.copy(); bad[TASKS[0]] = bad[TASKS[0]].astype(float); bad.loc[0, TASKS[0]] = value
            bad.to_csv(self.csv, index=False)
            with self.assertRaises(ValueError):
                load_table(self.csv, 0)

    def test_missing_subject_and_duplicate_images_fail(self):
        original = pd.read_csv(self.csv)
        bad = original.copy(); bad.loc[0, "user"] = None; bad.to_csv(self.csv, index=False)
        with self.assertRaises(ValueError):
            load_table(self.csv, 0)
        original.to_csv(self.csv, index=False)
        (self.folder / original.ID.iloc[1]).write_bytes((self.folder / original.ID.iloc[0]).read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_table(self.csv, 0)

    def test_group_overlap_is_rejected(self):
        df = load_table(self.csv, 0)
        df["acquisition"] = "shared_scene"
        with self.assertRaisesRegex(ValueError, "shared acquisition"):
            make_splits(df, "loso", "user", 42, extra_group_cols=["acquisition"])

    def test_missing_and_changed_images_fail(self):
        df = load_table(self.csv, 0)
        dataset = PostureDataset(df, 8)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(df.iloc[0]._path)
        with self.assertRaisesRegex(ValueError, "changed"):
            dataset[0]
        Path(df.iloc[0]._path).unlink()
        with self.assertRaises(FileNotFoundError):
            load_table(self.csv, 0)

    @unittest.skipUnless(os.environ.get("SPECTRE_TEST_WORKERS") == "1", "Optional multiprocessing test: set SPECTRE_TEST_WORKERS=1 outside restricted sandboxes")
    def test_worker_loader_keeps_order_and_samples(self):
        dataset = PostureDataset(load_table(self.csv, 0), 8)
        expected = list(loader(dataset, 5, 0, 42))
        actual = list(loader(dataset, 5, 1, 42))
        self.assertEqual(sum(len(labels) for _, labels in actual), 18)
        for (image_a, label_a), (image_b, label_b) in zip(expected, actual):
            torch.testing.assert_close(image_a, image_b)
            torch.testing.assert_close(label_a, label_b)

    def test_patch_order_and_singleton_batch(self):
        image = torch.arange(1*3*8*8).reshape(1, 3, 8, 8)
        actual = Patchify(4, 4)(image)
        expected = torch.stack([image[0, :, y:y+4, x:x+4].permute(1, 2, 0).flatten() for y in [0, 4] for x in [0, 4]])
        torch.testing.assert_close(actual[0], expected)
        model = tiny_model()
        self.assertIn("pos_embedding", dict(model.named_buffers()))
        data = TensorDataset(torch.rand(3, 3, 8, 8), torch.zeros(3, 8, dtype=torch.long))
        for batch_size in [1, 2, 3]:
            true, pred = predict(model, loader(data, batch_size, 0, 1), "cpu")
            self.assertEqual(true.shape, (3, 8))
            self.assertEqual(pred.shape, (3, 8))

    def test_loss_is_sample_weighted(self):
        class IdentityHeads(torch.nn.Module):
            def forward(self, x):
                return [x]
        logits = torch.tensor([[2., 0.], [2., 0.], [0., 4.]])
        targets = torch.zeros((3, 1), dtype=torch.long)
        expected = torch.nn.functional.cross_entropy(logits, targets[:, 0]).item()
        for batch in [1, 2, 3]:
            loss, heads = epoch_loss(IdentityHeads(), loader(TensorDataset(logits, targets), batch, 0, 1), "cpu")
            self.assertAlmostEqual(loss, expected, places=6)
            self.assertAlmostEqual(heads[0], expected, places=6)
        with self.assertRaisesRegex(ValueError, "Empty"):
            epoch_loss(IdentityHeads(), [], "cpu")

    def test_external_labels_cannot_change_selected_weights(self):
        original = load_table(self.csv, 0)
        split = make_splits(original, "loso", "user", 42)[0]["indices"]
        changed = original.copy()
        for task, count in zip(TASKS, DIMS):
            changed.loc[split["test"], task] = (changed.loc[split["test"], task] + 1) % count
        weights, selections = [], []
        for i, df in enumerate([original, changed]):
            set_seed(43)
            model = tiny_model()
            train = loader(PostureDataset(df.iloc[split["train"]], 8), 2, 0, 43, True)
            val = loader(PostureDataset(df.iloc[split["validation"]], 8), 2, 0, 43)
            selections.append(fit(model, train, val, "cpu", 2, 2, .01, self.folder/f"fit{i}"))
            weights.append({k: v.clone() for k, v in model.state_dict().items()})
        self.assertEqual(selections[0], selections[1])
        for key in weights[0]:
            torch.testing.assert_close(weights[0][key], weights[1][key], rtol=0, atol=0)

    def test_fixed_class_metrics_and_single_fold_sd(self):
        y = np.zeros((3, 8), dtype=int)
        rows = export_metrics(y, y, self.folder / "metrics")
        self.assertEqual(rows[0]["accuracy"], 1.)
        self.assertAlmostEqual(rows[0]["f1_macro"], 1/6)
        self.assertEqual(rows[0]["balanced_accuracy"], 1.)
        export_fold_summary([{**row, "fold": 1} for row in rows], self.folder, False)
        summary = json.loads((self.folder / "fold_summary.json").read_text())
        self.assertIsNone(summary[0]["accuracy_sd"])
        self.assertFalse(summary[0]["complete_protocol"])

    def test_end_to_end_checkpoint_and_evaluation_guards(self):
        run = self.folder / "run"
        arguments = ["--protocol", "loso", "--csv_path", str(self.csv), "--label_base", "0",
                     "--epochs", "1", "--batch_size", "2", "--image_size", "8", "--patch_size", "4",
                     "--dim", "8", "--depth", "1", "--heads", "1", "--dim_head", "8", "--mlp_dim", "16", "--device", "cpu"]
        main(arguments + ["--output_dir", str(run)])
        predictions = pd.read_csv(run / "predictions.csv")
        self.assertEqual(len(predictions), 18)
        self.assertFalse(predictions.ID.duplicated().any())
        self.assertTrue(json.loads((run / "completion.json").read_text())["complete_protocol"])
        with self.assertRaises(FileExistsError):
            main(arguments + ["--output_dir", str(run)])
        raw = pd.read_csv(self.csv).iloc[::-1]; raw.to_csv(self.csv, index=False)
        evaluated = self.folder / "eval"
        main(arguments + ["--output_dir", str(evaluated), "--eval_only", "--weights_dir", str(run)])
        pd.testing.assert_frame_equal(predictions, pd.read_csv(evaluated / "predictions.csv"))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            checkpoint_preflight(run / "fold_1", "wrong")
        (run / "fold_6" / "best.pt").unlink()
        rejected = self.folder / "missing"
        with self.assertRaises(FileNotFoundError):
            main(arguments + ["--output_dir", str(rejected), "--eval_only", "--weights_dir", str(run)])
        self.assertFalse(rejected.exists())

    def test_group5_and_standalone_fold_match(self):
        arguments = ["--protocol", "group5", "--csv_path", str(self.csv), "--label_base", "0",
                     "--epochs", "1", "--batch_size", "2", "--image_size", "8", "--patch_size", "4",
                     "--dim", "8", "--depth", "1", "--heads", "1", "--dim_head", "8", "--mlp_dim", "16", "--device", "cpu"]
        full, single = self.folder / "group5", self.folder / "single"
        main(arguments + ["--output_dir", str(full)])
        main(arguments + ["--output_dir", str(single), "--fold", "3"])
        first = torch.load(full / "fold_3/best.pt", weights_only=True)
        second = torch.load(single / "fold_3/best.pt", weights_only=True)
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        self.assertEqual(len(pd.read_csv(full / "predictions.csv")), 18)
        self.assertFalse(json.loads((single / "completion.json").read_text())["complete_protocol"])
        self.assertFalse((single / "pooled_sample_metrics").exists())
        with self.assertRaises(ValueError):
            main(arguments + ["--output_dir", str(self.folder / "bad"), "--fold", "6"])

    def test_checkpoint_tampering_and_changed_targets_are_rejected(self):
        from spectre.cli import sha256_file
        from spectre.engine import write_json
        weights = self.folder / "best.pt"
        weights.write_bytes(b"example")
        write_json(self.folder / "checkpoint.json", {"signature": "expected", "weights_sha256": sha256_file(weights)})
        checkpoint_preflight(self.folder, "expected")
        weights.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            checkpoint_preflight(self.folder, "expected")
        df = load_table(self.csv, 0)
        before = digest(records(df))
        df.loc[0, TASKS[0]] = 1
        self.assertNotEqual(before, digest(records(df)))

    def test_empty_missing_duplicate_schema_and_small_cohort(self):
        df = load_table(self.csv, 0)
        with self.assertRaises(ValueError):
            make_splits(df[df.user.isin(["S0", "S1"])], "loso", "user", 42)
        with self.assertRaises(ValueError):
            make_splits(df[df.user.isin(["S0", "S1", "S2", "S3"])], "group5", "user", 42)
        original = self.csv.read_text()
        self.csv.write_text(original.replace("ID,user", "ID,ID", 1))
        with self.assertRaisesRegex(ValueError, "Duplicate CSV"):
            load_table(self.csv, 0)
        self.csv.write_text("ID,user\n")
        with self.assertRaisesRegex(ValueError, "Missing CSV"):
            load_table(self.csv, 0)


if __name__ == "__main__":
    unittest.main()
