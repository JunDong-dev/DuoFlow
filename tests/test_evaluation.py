"""CPU tests for BC aggregation, root transforms, and held-out entry points."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate
import preprocess
from data_splits import claim_preprocessing_scope


class RotationConversions:
    @staticmethod
    def rotation_6d_to_matrix(value):
        first = torch.nn.functional.normalize(value[..., :3], dim=-1)
        second = value[..., 3:]
        second = torch.nn.functional.normalize(second - (first * second).sum(-1, keepdim=True) * first, dim=-1)
        return torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-2)

    @staticmethod
    def matrix_to_axis_angle(value):
        shape = value.shape[:-2]
        array = value.detach().cpu().numpy().reshape(-1, 3, 3)
        return torch.from_numpy(Rotation.from_matrix(array).as_rotvec()).float().reshape(*shape, 3)


def motion_fixture():
    angles = np.array([[0, 0, 0], [0, np.pi / 2, 0], [0, np.pi, 0]], dtype=np.float32)
    matrices = np.broadcast_to(np.eye(3), (3, 55, 3, 3)).copy()
    matrices[:, 0] = Rotation.from_rotvec(angles).as_matrix()
    translation = np.array([[0, 0, 0], [0.2, 0, 0.1], [0.4, 0, 0.2]], dtype=np.float32)
    return np.concatenate((translation, matrices[:, :, :2, :].reshape(3, 330)), axis=1).astype(np.float32)


class FakeBody:
    def __call__(self, **kwargs):
        local = np.zeros((len(kwargs["transl"]), 55, 3), dtype=np.float32)
        local[:, [7, 8, 10, 11], 0] = 1
        rotation = Rotation.from_rotvec(kwargs["global_orient"].cpu().numpy()).as_matrix()
        joints = np.einsum("tij,tkj->tki", rotation, local) + kwargs["transl"].cpu().numpy()[:, None, :]
        return SimpleNamespace(joints=torch.from_numpy(joints).float())


def backend(body):
    value = evaluate.MotionBackend.__new__(evaluate.MotionBackend)
    value.device = torch.device("cpu")
    value.fk_chunk_size = 2
    value.rc = RotationConversions
    value.smplx = body
    return value


class EvaluationTests(unittest.TestCase):
    def test_bc_gap_uses_reference_audio_and_absolute_after_mean(self):
        rows = [{"metrics": {"bc_gt_audio_pred_motion": pred, "bc_gt_audio_gt_motion": gt,
                             "bc_generated_audio_pred_motion": 0.99}} for pred, gt in ((0.8, 0.2), (0.4, 0.8))]
        summary = evaluate.summarize_group(rows, [], [], None)
        self.assertAlmostEqual(summary["bc_gap"], 0.1)
        self.assertEqual(summary["bc_gap"], summary["bc_gt_audio_pred_motion_gap_from_human_abs"])
        self.assertIsNone(evaluate.summarize_group([], [], [], None)["bc_gap"])

    def test_world_joints_include_root_rotation_and_translation_once(self):
        motion = motion_fixture()
        model = backend(FakeBody())
        canonical = model.positions(motion)
        world = model.positions(motion, world_space=True)
        np.testing.assert_allclose(canonical[:, 7], [[1, 0, 0]] * 3, atol=1e-6)
        np.testing.assert_allclose(world[:, 7], [[1, 0, 0], [0.2, 0, -0.9], [-0.6, 0, 0.2]], atol=1e-6)
        self.assertEqual(evaluate.foot_skating(canonical, 30)["skating_ratio"], 0)
        self.assertEqual(evaluate.foot_skating(world, 30)["skating_ratio"], 1)

    def test_static_world_feet_do_not_skate(self):
        world = np.zeros((5, 55, 3), dtype=np.float32)
        self.assertEqual(evaluate.foot_skating(world, 30)["skating_ratio"], 0)
        world[:, :, 0] = np.arange(5)[:, None] * 0.2
        self.assertEqual(evaluate.foot_skating(world, 30)["skating_ratio"], 1)

    def test_default_evaluation_uses_test_split(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", ["evaluate.py"]):
            args = evaluate.parse_args()
        self.assertEqual(args.split, "test")
        self.assertEqual(args.gt_motion_dir, evaluate.DEFAULT_GT_MOTION_DIR)
        self.assertEqual(args.input_dir, evaluate.DEFAULT_INPUT_DIR)

    def test_batch_inference_defaults_follow_the_selected_split(self):
        import infer_batch
        assets = ["--trained-model", "model.safetensors", "--text-encoder-path", "gemma",
                  "--audioldm2-cache-dir", "audio", "--phoneme-encoder-path", "phoneme",
                  "--phoneme-tokenizer-path", "tokenizer"]
        with patch.dict(os.environ, {}, clear=True):
            for split in ("test", "val"):
                with patch.object(sys, "argv", ["infer_batch.py", "--split", split, *assets]):
                    args = infer_batch.parse_args()
                self.assertEqual(args.l1_json.name, f"{split}_l1.json")
                self.assertEqual(args.l2_json.name, f"{split}_l2.json")
                self.assertEqual(args.val_root.name, split)
                self.assertEqual(args.output_dir.name, split)
                self.assertEqual(args.num_inference_steps, 30)
                self.assertEqual((args.motion_guidance_scale, args.audio_guidance_scale), (1, 5))
                self.assertEqual(args.seeds, (0,))

    def test_single_inference_defaults_use_modality_cfg(self):
        import infer
        argv = ["infer.py", "--trained-model-path", "model", "--text-encoder-path", "gemma",
                "--motion-vae-path", "vae", "--prompt", "<S>Hello.</E>", "--output-path", "output.mp4"]
        with patch.object(sys, "argv", argv):
            args = infer.parse_args()
        self.assertEqual(args.num_inference_steps, 30)
        self.assertEqual((args.cfg_guidance_scale_motion, args.cfg_guidance_scale_audio), (1, 5))
        self.assertEqual(args.seed, 0)

    def test_training_cache_cannot_be_evaluated_as_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            claim_preprocessing_scope(root, root / "raw", "train", False)
            with patch.object(sys, "argv", ["evaluate.py", "--gt-motion-dir", str(root / "gt_motion")]):
                with self.assertRaises(ValueError):
                    evaluate.parse_args()

    def test_dataset_filters_ids_and_invalidates_old_inventory(self):
        from motionspeech_trainer.datasets import PrecomputedDataset
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for source in ("latents", "conditions"):
                (root / source).mkdir()
                for sample_id in ("one_l1", "one_l2", "two_l1", "two_l2"):
                    (root / source / f"{sample_id}.pt").touch()
            ids = root / "sample_ids.txt"
            ids.write_text("one\n")
            first = PrecomputedDataset(str(root), data_sources=["latents", "conditions"])
            self.assertEqual(len(first), 2)
            self.assertTrue(all(path.stem.startswith("one_") for path in first.sample_files["latents"]))
            ids.write_text("two\n")
            second = PrecomputedDataset(str(root), data_sources=["latents", "conditions"])
            self.assertEqual(len(second), 2)
            self.assertTrue(all(path.stem.startswith("two_") for path in second.sample_files["latents"]))

    def test_audio_only_preprocessing_needs_no_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for directory, suffix in (("audio_wav", ".wav"), ("caption", ".txt")):
                (root / directory).mkdir()
                (root / directory / ("one" + suffix)).touch()
            preprocess.validate_raw_files(root, ["one"], audio_only=True)
            with self.assertRaises(FileNotFoundError):
                preprocess.validate_raw_files(root, ["one"])
            argv = ["preprocess.py", "--data-root", str(root), "--output-root", str(root / "out"),
                    "--text-encoder", "model", "--audio-only"]
            with patch.object(sys, "argv", argv):
                args = preprocess.parse_args()
            self.assertNotIn("motion", args.stages)
            self.assertEqual(args.variants, ("speech",))
            captions = {"one": "<S>Hello.</E><AUDCAP>Calm.</AUDCAP>"}
            with patch.object(preprocess, "load_text_encoder", return_value=object()), \
                 patch.object(preprocess, "save_text_condition") as save:
                preprocess.preprocess_text(args, ["one"], captions)
            self.assertEqual(save.call_count, 1)
            self.assertEqual(save.call_args.args[2].name, "one_speech.pt")
            self.assertFalse((args.output_root / "joint_conditions").exists())
            preprocess.write_manifests(args, ["one"], captions)
            rows = json.loads((args.output_root / "manifests/train_speech.json").read_text())
            self.assertNotIn("motion_path", rows[0])

    @unittest.skipUnless(os.environ.get("DUOFLOW_TEST_SMPLX_DIR"), "Set DUOFLOW_TEST_SMPLX_DIR for the licensed SMPL-X fixture")
    def test_actual_smplx_world_fk_matches_rigid_root_transform(self):
        import smplx
        body = smplx.create(os.environ["DUOFLOW_TEST_SMPLX_DIR"], model_type="smplx", gender="NEUTRAL_2020",
                            use_face_contour=False, num_betas=300, num_expression_coeffs=100, ext="npz", use_pca=False).eval()
        model = backend(body)
        motion = motion_fixture()
        canonical = model.positions(motion)
        world = model.positions(motion, world_space=True)
        rotation = Rotation.from_rotvec([[0, 0, 0], [0, np.pi / 2, 0], [0, np.pi, 0]]).as_matrix()
        pelvis = canonical[:, :1]
        expected = np.einsum("tij,tkj->tki", rotation, canonical - pelvis) + pelvis + motion[:, None, :3]
        np.testing.assert_allclose(world, expected, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
