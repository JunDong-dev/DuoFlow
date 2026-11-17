# Modified for DuoFlow; see NOTICE.md.
"""CPU-only multi-tap CTC regression tests; run with unittest."""
import ast
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass, replace

import torch
from safetensors.torch import save_file, load_file
from motionspeech_trainer.config import ModelConfig
from motionspeech_trainer.training_strategies.text_to_motion_audio import TextToMotionAudioStrategy
from motionspeech_core.model.transformer.model import MotionSpeechModel


def strategy(taps: list[int]) -> TextToMotionAudioStrategy:
    obj = TextToMotionAudioStrategy.__new__(TextToMotionAudioStrategy)
    torch.nn.Module.__init__(obj)
    obj.config = SimpleNamespace(aux_ctc_weight=0.1, aux_ctc_hidden_dim=8, aux_ctc_num_classes=5)
    obj.audio_ctc_head = torch.nn.Linear(8, 5)
    obj.audio_ctc_heads = torch.nn.ModuleDict()
    obj.probe = torch.nn.Linear(1, 1)
    obj.build_per_tap_ctc_heads(taps)
    return obj


def inputs(hidden: tuple[torch.Tensor, ...], taps: list[int]) -> SimpleNamespace:
    return SimpleNamespace(ctc_enabled=True, audio_hidden_aux=hidden[-1],
        audio_hidden_aux_multi=hidden if len(hidden)>1 else None,
        audio_hidden_aux_tap_blocks=tuple(taps),
        audio_loss_mask=torch.ones(2, 8, dtype=torch.bool),
        phoneme_ids=torch.tensor([[1, 2, 0], [3, 1, 2]]),
        audio=SimpleNamespace(phoneme_context_mask=torch.ones(2, 3)))


class MultiTapTests(unittest.TestCase):
    def test_config(self) -> None:
        cfg = ModelConfig(num_transformer_layers=8, aux_ctc_tap_blocks=[7, 0, 3, 3])
        self.assertEqual(cfg.aux_ctc_tap_blocks, [0, 3, 7])
        self.assertIsNone(ModelConfig(aux_ctc_tap_blocks=[]).aux_ctc_tap_blocks)
        for taps in ([-1], [8]):
            with self.assertRaises(ValueError):
                ModelConfig(num_transformer_layers=8, aux_ctc_tap_blocks=taps)

    def test_average_and_gradients(self) -> None:
        taps = list(range(8))
        obj = strategy(taps)
        self.assertIsNone(obj.audio_ctc_head)
        hidden = tuple(torch.randn(2, 8, 8, requires_grad=True) for _ in taps)
        data = inputs(hidden, taps)
        loss = obj._compute_ctc_aux_loss(data)
        separate = []
        for index, h in zip(taps, hidden):
            separate.append(torch.nn.functional.ctc_loss(
                obj.audio_ctc_heads[str(index)](h).log_softmax(-1).transpose(0, 1),
                data.phoneme_ids, torch.tensor([8, 8]), torch.tensor([2, 3]),
                blank=4, zero_infinity=True))
        torch.testing.assert_close(loss, torch.stack(separate).mean())
        loss.backward()
        self.assertTrue(all(h.grad is not None and torch.isfinite(h.grad).all() for h in hidden))
        self.assertTrue(all(h.weight.grad is not None for h in obj.audio_ctc_heads.values()))
        data.ctc_enabled = False
        self.assertIsNone(obj._compute_ctc_aux_loss(data))
        data.ctc_enabled = True
        data.audio_hidden_aux = None
        self.assertIsNone(obj._compute_ctc_aux_loss(data))

    def test_single_tap(self) -> None:
        obj = strategy([3])
        hidden = (torch.randn(2, 8, 8, requires_grad=True),)
        loss = obj._compute_ctc_aux_loss(inputs(hidden, [3]))
        loss.backward()
        self.assertIsNotNone(obj.audio_ctc_head.weight.grad)

    def test_model_taps_and_checkpoint_recompute(self) -> None:
        @dataclass
        class Args:
            x: torch.Tensor

        class Block(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = torch.nn.Linear(8, 8)

            def forward(self, video: Args, audio: Args, perturbations: object) -> tuple[Args, Args]:
                return video, replace(audio, x=audio.x + self.proj(audio.x))

        for checkpointing in (False, True):
            model = MotionSpeechModel(num_attention_heads=1, attention_head_dim=8,
                in_channels=8, out_channels=8, num_layers=8, caption_channels=8,
                cross_attention_dim=8, audio_num_attention_heads=1, audio_attention_head_dim=8,
                audio_in_channels=8, audio_out_channels=8, audio_cross_attention_dim=8,
                aux_ctc_tap_blocks=list(range(8)))
            model.transformer_blocks = torch.nn.ModuleList([Block() for _ in range(8)])
            model.set_gradient_checkpointing(checkpointing)
            x = Args(torch.randn(2, 8, 8, requires_grad=True))
            model._process_transformer_blocks(x, x, None)
            self.assertEqual(sorted(model.aux_audio_hiddens), list(range(8)))
            heads = strategy(list(range(8)))
            loss = heads._compute_ctc_aux_loss(inputs(tuple(model.aux_audio_hiddens.values()), list(range(8))))
            loss.backward()
            self.assertTrue(all(b.proj.weight.grad is not None for b in model.transformer_blocks))
            model.aux_ctc_tap_blocks = [3]
            model._process_transformer_blocks(x, x, None)
            self.assertEqual(sorted(model.aux_audio_hiddens), [3])

    def test_checkpoint_transition_and_strictness(self) -> None:
        # Exercise the production loader without constructing GPU trainer models.
        path = Path(__file__).resolve().parents[1] / 'src/motionspeech_trainer/trainer.py'
        tree = ast.parse(path.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_load_full_checkpoint')
        namespace = dict(Path=Path, torch=torch, load_file=load_file, logger=logging.getLogger('ctc_test'))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), 'exec'), namespace)
        model = torch.nn.Linear(2, 2)
        old = strategy([3])
        new = strategy(list(range(8)))
        trainer = SimpleNamespace(_transformer=model, _training_strategy=new,
            _config=SimpleNamespace(model=SimpleNamespace(allow_missing_checkpoint_keys=False)))
        def load(state: dict[str, torch.Tensor]) -> None:
            with tempfile.TemporaryDirectory() as folder:
                checkpoint = Path(folder)/'test.safetensors'
                save_file(state, checkpoint)
                namespace['_load_full_checkpoint'](trainer, checkpoint)
        state = dict(model.state_dict())
        state.update({'strategy.'+k:v for k,v in old.state_dict().items()})
        load(state)
        torch.testing.assert_close(old.probe.weight, new.probe.weight)
        # The current stage2 checkpoint has no CTC head parameters at all.
        load({k: v for k, v in state.items() if 'audio_ctc_head.' not in k})
        multi = dict(model.state_dict())
        multi.update({'strategy.'+k:v.clone() for k,v in new.state_dict().items()})
        trainer._training_strategy = strategy(list(range(8)))
        load(multi)
        for k,v in new.state_dict().items():
            torch.testing.assert_close(v, trainer._training_strategy.state_dict()[k])
        del state['strategy.probe.weight']
        with self.assertRaises(RuntimeError):
            load(state)


if __name__ == '__main__':
    unittest.main()
