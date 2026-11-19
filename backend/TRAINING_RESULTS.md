# Independent AI Lab — consolidated training record

## v0.1

- Randomly initialized compact Transformer.
- CPU training: 600 steps.
- Parameters: 124,160.
- Initial validation next-byte loss: 5.53476.
- Final validation next-byte loss: 2.27737.
- Checkpoint preserved in `checkpoints/independent_lm_v0_1.pt`.

## v0.2

- Fresh random initialization; did not load v0.1 or third-party pretrained weights.
- 377,568 parameters; 3 layers; width 96; 4 heads; 192-byte context.
- 1,400 CPU optimizer steps on the v0.1 seed text plus 138 authored educational Q&A documents.
- Initial sampled validation loss: 5.6016.
- Final sampled validation loss: 2.27727; best sampled validation loss: 2.24975.
- Independent held-out sequential-window loss: 2.27219; byte perplexity: 9.7006.

## v0.3 dataset build

- 71 project-authored Q&A documents.
- 71 instruction/response JSONL examples.
- Dataset manifest records provenance and synthetic-corpus limitations.
- No model optimization step occurred in v0.3 itself.

## v0.4 continued language-model training

- Starting point: this project's own v0.2 checkpoint.
- 400 additional CPU optimizer steps.
- 71 new v0.3 documents; 64 train and 7 held out in this new split. The prior v0.2 held-out set was also preserved.
- Train/validation bytes: 146,339 / 7,365.
- Initial sampled train/validation loss: 2.25097 / 2.23644.
- Final sampled train/validation loss: 2.25475 / 2.23302.
- Best sampled validation loss: 2.22887.
- Checkpoint: `checkpoints/independent_lm_v0_4.pt`.

## v0.5 instruction tuning

- Fine-tuned from v0.4; 180 steps; CPU.
- 64 training examples; 7 held out.
- Loss applied to answer bytes only.
- Initial held-out answer loss: 2.36913.
- Final held-out answer loss: 2.36323.
- Small improvement on a tiny synthetic held-out set; generalized instruction-following is not established.
- Checkpoint: `checkpoints/independent_lm_v0_5.pt`.

## v0.6 synthetic diffusion experiment

- Random initialization; no pretrained image model or external API.
- Toy conditional U-Net: 303,299 parameters.
- 192 procedurally generated 32×32 images; six geometric classes.
- 240 CPU optimizer steps.
- Noise-prediction MSE: 0.21850 at step 60; 0.15420 at 120; 0.12610 at 180; 0.10284 at 240.
- Sample quality is still visibly noisy; the model is not useful general-purpose image generation.
- Checkpoint: `checkpoints/image_diffusion_toy_v0_6.pt`.

## Software verification

- Automated tests: 11 passed.
- FastAPI smoke tests: `/api/health` returned ready; Q&A retrieval returned a matched local corpus answer; image endpoint returned an encoded PNG.
- No external AI API calls are made by the app.

## What the evidence supports

The project demonstrates an independently initialized small language-model experiment, subsequent continued training, a limited instruction-tuning run, a toy diffusion training pipeline, local retrieval/memory, and an integrated local application.

## What the evidence does not support

It does not show ChatGPT-level intelligence, frontier performance, broad reasoning ability, robust general instruction following, good image synthesis, image understanding, speech, video, or a novel neural architecture. The included model sizes and datasets are far below what would be required to compete with frontier systems.
