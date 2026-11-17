# DuoFlow

### Towards Synchronized Speech and Motion Generation with Fine-Grained Control

Official PyTorch implementation of **DuoFlow** for text-driven joint speech and 3D co-speech motion generation.

[Paper](https://arxiv.org/abs/2611.21530) · [Project page](https://jundong-dev.github.io/DuoFlow-project-page/) · [Dataset page](https://huggingface.co/datasets/JunDong-dev/motionspeech)

---

Navigation: [Release status](#open-source-release-status) · [Installation](#installation) · [Data preparation](#data-preparation) · [Training](#training) · [Inference](#inference) · [Evaluation](#evaluation) · [Citation](#citation)

## Open-source release status

| Release | Status | Access |
| :-- | :-- | :-- |
| Paper | ✅ Available | [arXiv:2611.21530](https://arxiv.org/abs/2611.21530) |
| Training, inference and evaluation code | ✅ Available | [GitHub](https://github.com/JunDong-dev/DuoFlow) · [Source files](scripts/) · [Configurations](configs/) |
| Project page and video demos | ✅ Available | [Project page](https://jundong-dev.github.io/DuoFlow-project-page/) |
| MotionSpeech dataset | ⏳ Data files pending | [Hugging Face](https://huggingface.co/datasets/JunDong-dev/motionspeech) |
| Pretrained DuoFlow, Motion VAE and PairAcc checkpoints | ⏳ Download links pending | — |

## Installation

Use a Linux machine with an NVIDIA GPU and Python 3.11. Run the following commands from the repository root with `uv` installed:

```bash
sudo apt-get update
sudo apt-get install -y espeak-ng ffmpeg

uv sync --python 3.11
source .venv/bin/activate
```

The workspace contains `motionspeech-core` and `motionspeech-trainer`; `uv sync` installs both packages and their dependencies. Use a CUDA-enabled PyTorch installation compatible with your GPU driver.

## Data preparation

### MotionSpeech data

Place the dataset under the following directory structure, preserving its official train, validation and test splits:

```text
motionspeech_data_opensource/
├── audio_wav/
│   └── <sample_id>.wav
├── caption/
│   └── <sample_id>.txt
├── motion_smplx/
│   └── <sample_id>.npz
└── split/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Captions contain the motion conditions, speech transcript and voice description. The preprocessing scripts read the transcript from `<S>...</E>`.

### Required resources

Download the external resources needed for your workflow and set their local paths below. Model checkpoints and SMPL-X body-model files are supplied separately from the source code.

| Resource | Purpose | Source |
| :-- | :-- | :-- |
| Gemma 3 12B IT | Text conditioning | [Google](https://huggingface.co/google/gemma-3-12b-it) |
| AudioLDM2 GigaSpeech | Audio VAE, vocoder and phoneme-conditioning resources | [Hugging Face](https://huggingface.co/anhnct/audioldm2_gigaspeech) |
| Motion VAE checkpoint | Motion encoding and decoding | Train using the instructions below |
| PantoMatrix / EMAGE | Motion evaluation | [PantoMatrix](https://github.com/PantoMatrix/PantoMatrix) |
| EMAGE evaluation assets | FGD encoder and evaluation statistics | [Hugging Face](https://huggingface.co/H-Liu1997/emage_evaltools) |
| SMPL-X body models | Motion conversion and evaluation | [SMPL-X](https://smpl-x.is.tue.mpg.de/) |
| Whisper large-v3-turbo | Speech recognition for WER | [Hugging Face](https://huggingface.co/openai/whisper-large-v3-turbo) |
| WavLM base-plus-SV | Speaker similarity | [Hugging Face](https://huggingface.co/microsoft/wavlm-base-plus-sv) |
| MotionSpeech-adapted CLaM-H checkpoint | L1/L2 Global PairAcc | [Evaluator requirements](third_party/clam_h/README.md) · [Upstream CLaM-H](https://github.com/SheldongChen/CLaM-H) |

```bash
export MOTIONSPEECH_DATA_ROOT=/path/to/motionspeech_data_opensource
export MOTIONSPEECH_TEXT_ENCODER=/path/to/gemma-3-12b-it
export MOTIONSPEECH_AUDIOLDM2_CACHE=/path/to/huggingface/hub
export MOTIONSPEECH_PHONEME_ENCODER=/path/to/audioldm2_snapshot/text_encoder_2
export MOTIONSPEECH_PHONEME_TOKENIZER=/path/to/audioldm2_snapshot/tokenizer_2

export DUOFLOW_MOTION_VAE=/path/to/motion_vae.pth
export MOTIONSPEECH_EMAGE_ROOT=/path/to/PantoMatrix
export MOTIONSPEECH_ASR_MODEL=/path/to/whisper-large-v3-turbo
export MOTIONSPEECH_SPEAKER_MODEL=/path/to/wavlm-base-plus-sv
export DUOFLOW_PAIRACC_CHECKPOINT=/path/to/adapted_clam_h/best.tar
export DUOFLOW_PAIRACC_MODEL_ROOT=/path/to/smpl_models
```

Keep a complete AudioLDM2 GigaSpeech snapshot in `MOTIONSPEECH_AUDIOLDM2_CACHE`. The phoneme encoder and tokenizer paths should point to `text_encoder_2/` and `tokenizer_2/` in that snapshot.

Place the EMAGE assets at:

```text
$MOTIONSPEECH_EMAGE_ROOT/emage_evaltools/
├── AESKConv_240_100.bin
├── mean_vel_smplxflame_30.npy
└── smplx_models/smplx/SMPLX_NEUTRAL_2020.npz
```

PairAcc additionally expects `$DUOFLOW_PAIRACC_MODEL_ROOT/smplx/SMPLX_NEUTRAL.npz` and a MotionSpeech-adapted evaluator checkpoint.

### Prepare the Motion VAE

If a compatible Motion VAE checkpoint is available, set `DUOFLOW_MOTION_VAE` to its path. To train one from the raw motion data:

```bash
python scripts/train_motion_vae.py \
  --data-root "$MOTIONSPEECH_DATA_ROOT" \
  --split train \
  --config configs/motion_vae.yaml \
  --output-dir outputs/motion_vae \
  --device cuda:0

export DUOFLOW_MOTION_VAE="$PWD/outputs/motion_vae/checkpoint_last.pth"
```

Reconstruct motion with the trained checkpoint:

```bash
python scripts/infer_motion_vae.py \
  --data-root "$MOTIONSPEECH_DATA_ROOT" \
  --split test \
  --checkpoint "$DUOFLOW_MOTION_VAE" \
  --mode reconstruct \
  --output-dir outputs/motion_vae_reconstruction \
  --device cuda:0
```

Use `--mode encode` to save motion latents, or `--mode decode --latent-dir /path/to/latents` to decode an existing latent cache.

### Preprocess MotionSpeech

Create a separate latent cache for each split:

```bash
export MOTIONSPEECH_TRAIN_ROOT="$PWD/outputs/preprocessed/train"
export MOTIONSPEECH_VAL_ROOT="$PWD/outputs/preprocessed/val"
export MOTIONSPEECH_TEST_ROOT="$PWD/outputs/preprocessed/test"

for split in train val test; do
  python scripts/preprocess.py \
    --data-root "$MOTIONSPEECH_DATA_ROOT" \
    --split "$split" \
    --output-root "$PWD/outputs/preprocessed/$split" \
    --text-encoder "$MOTIONSPEECH_TEXT_ENCODER" \
    --audioldm2-cache "$MOTIONSPEECH_AUDIOLDM2_CACHE" \
    --phoneme-tokenizer "$MOTIONSPEECH_PHONEME_TOKENIZER" \
    --motion-vae-checkpoint "$DUOFLOW_MOTION_VAE" \
    --device cuda:0
done
```

Each output root contains motion and audio latents, text features, phoneme IDs, ground-truth motion, and `manifests/<split>_{l1,l2}.json` for the corresponding split. Use the training cache for optimization, the validation cache for model selection, and the test cache for evaluation.

### Prepare speech-pretraining data

Stage 1 uses Emilia speech data. Arrange the selected speech data with `audio_wav/`, `caption/`, and `split/{train,val}.txt`. Captions must include `<S>...</E>` and may include `<AUDCAP>...</AUDCAP>`.

```bash
export EMILIA_DATA_ROOT=/path/to/emilia_audio_subset
export EMILIA_TRAIN_ROOT="$PWD/outputs/emilia/train"
export EMILIA_VAL_ROOT="$PWD/outputs/emilia/val"

for split in train val; do
  python scripts/preprocess.py \
    --data-root "$EMILIA_DATA_ROOT" \
    --audio-only \
    --split "$split" \
    --output-root "$PWD/outputs/emilia/$split" \
    --text-encoder "$MOTIONSPEECH_TEXT_ENCODER" \
    --audioldm2-cache "$MOTIONSPEECH_AUDIOLDM2_CACHE" \
    --phoneme-tokenizer "$MOTIONSPEECH_PHONEME_TOKENIZER" \
    --device cuda:0
done
```

## Training

### Stage 1: speech pretraining

```bash
export MOTIONSPEECH_STAGE1_OUTPUT="$PWD/outputs/stage1"
python scripts/train.py configs/stage1.yaml
```

### Stage 2: joint speech-motion training

Initialize Stage 2 from a Stage 1 checkpoint:

```bash
export MOTIONSPEECH_STAGE1_CHECKPOINT="$MOTIONSPEECH_STAGE1_OUTPUT/checkpoints/best/model_weights_best.safetensors"
export MOTIONSPEECH_STAGE2_OUTPUT="$PWD/outputs/stage2"

python scripts/train.py configs/stage2.yaml
```

The training settings are defined in [`configs/stage1.yaml`](configs/stage1.yaml) and [`configs/stage2.yaml`](configs/stage2.yaml). Stage 2 uses the MotionSpeech training cache and samples speech-only replay batches from `EMILIA_TRAIN_ROOT`.

For eight-GPU training, use the supplied Accelerate configuration:

```bash
accelerate launch \
  --config_file packages/motionspeech-trainer/configs/accelerate/ddp.yaml \
  scripts/train.py configs/stage1.yaml

accelerate launch \
  --config_file packages/motionspeech-trainer/configs/accelerate/ddp.yaml \
  scripts/train.py configs/stage2.yaml
```

## Inference

Generate speech and motion from the held-out test conditions:

```bash
export DUOFLOW_CHECKPOINT="$MOTIONSPEECH_STAGE2_OUTPUT/checkpoints/best/model_weights_best.safetensors"

python scripts/infer_batch.py \
  --split test \
  --data-root "$MOTIONSPEECH_TEST_ROOT" \
  --trained-model "$DUOFLOW_CHECKPOINT" \
  --output-dir outputs/inference/test \
  --variants l1 l2 \
  --all-samples \
  --seeds 0 \
  --num-inference-steps 30 \
  --motion-guidance-scale 1 \
  --audio-guidance-scale 5 \
  --text-encoder-path "$MOTIONSPEECH_TEXT_ENCODER" \
  --motion-vae-path "$DUOFLOW_MOTION_VAE" \
  --audioldm2-cache-dir "$MOTIONSPEECH_AUDIOLDM2_CACHE" \
  --phoneme-encoder-path "$MOTIONSPEECH_PHONEME_ENCODER" \
  --phoneme-tokenizer-path "$MOTIONSPEECH_PHONEME_TOKENIZER" \
  --use-correct-motion-time-coordinates \
  --device cuda:0
```

Set `DUOFLOW_CHECKPOINT` to a released checkpoint or a checkpoint produced by your own training run. Use `--variants l1` or `--variants l2` to select one motion-conditioning level. For validation data, use `--split val --data-root "$MOTIONSPEECH_VAL_ROOT"`. Generation outputs and per-sample metadata are written under `--output-dir`.

## Evaluation

Evaluate the generated test samples:

```bash
python scripts/evaluate.py \
  --split test \
  --input-dir outputs/inference/test \
  --output-dir outputs/evaluation/test \
  --gt-motion-dir "$MOTIONSPEECH_TEST_ROOT/gt_motion" \
  --emage-root "$MOTIONSPEECH_EMAGE_ROOT" \
  --asr-model "$MOTIONSPEECH_ASR_MODEL" \
  --speaker-model "$MOTIONSPEECH_SPEAKER_MODEL" \
  --pairacc-checkpoint "$DUOFLOW_PAIRACC_CHECKPOINT" \
  --pairacc-model-root "$DUOFLOW_PAIRACC_MODEL_ROOT" \
  --require-speech-metrics \
  --require-pairacc \
  --device cuda:0
```

The evaluator reports WER, speaker similarity, FGD, BC, BC Gap, foot skating, and L1/L2 Global PairAcc. Results include `metrics_summary.json`, `metrics_summary.csv`, per-sample metrics and `pairacc.json`.

To run PairAcc separately:

```bash
python scripts/evaluate_pairacc.py \
  --input-root outputs/inference/test \
  --output outputs/evaluation/test/pairacc.json \
  --checkpoint "$DUOFLOW_PAIRACC_CHECKPOINT" \
  --model-root "$DUOFLOW_PAIRACC_MODEL_ROOT" \
  --variants l1 l2 \
  --device cuda:0
```

PairAcc requires at least two distinct captions for each conditioning level and seed. See the [evaluator documentation](third_party/clam_h/README.md) for the motion representation and scoring procedure.

## Repository layout

```text
configs/                          Motion VAE and two-stage training configurations
packages/motionspeech-core/        Model components and generation utilities
packages/motionspeech-trainer/     Training, data loading and checkpoint management
scripts/train_motion_vae.py        Motion VAE training
scripts/infer_motion_vae.py        Motion encoding, decoding and reconstruction
scripts/preprocess.py              Motion, speech, text and phoneme preprocessing
scripts/train.py                   Stage 1 and Stage 2 training
scripts/infer_batch.py             Batch generation from held-out conditions
scripts/evaluate.py                Joint speech-motion evaluation
scripts/evaluate_pairacc.py        Standalone L1/L2 Global PairAcc
assets/motion_vae_stats/           Motion normalization statistics
third_party/                      Motion VAE and CLaM-H dependencies
```

## Acknowledgements

This implementation builds on [LTX-2](https://github.com/Lightricks/LTX-2) and uses [Gemma](https://huggingface.co/google/gemma-3-12b-it), [AudioLDM2](https://huggingface.co/anhnct/audioldm2_gigaspeech), [PantoMatrix](https://github.com/PantoMatrix/PantoMatrix), [CLaM-H](https://github.com/SheldongChen/CLaM-H), [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo), and [WavLM](https://huggingface.co/microsoft/wavlm-base-plus-sv). We thank the LTX-2 team and the authors of these projects for making their code and resources publicly available.

## Citation

If you use DuoFlow or MotionSpeech in your research, please cite:

```bibtex
@misc{dong2026duoflow,
  title={DuoFlow: Towards Synchronized Speech and Motion Generation with Fine-Grained Control},
  author={Jun Dong and Ronghui Li and Zhongyuan Hu and Yunqi Zhao and Lijie Guo and Ruizhang Xu and Di Kang and Linchao Bao and Xiu Li},
  year={2026},
  eprint={2611.21530},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2611.21530}
}
```
