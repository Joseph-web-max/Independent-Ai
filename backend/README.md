# Independent AI Lab — v1.0 Research Preview

**A self-contained research project with an independently initialized and trained small language model, a small instruction-tuning stage, a tiny synthetic diffusion experiment, local Q&A retrieval, local conversation memory, and a browser UI.**

This release consolidates milestones v0.3 through v1.0 into one downloadable project. It preserves the v0.1 and v0.2 checkpoints and training records.

> **Simple project layout:** There is exactly one application frontend (`index.html`) and one application backend entry point (`main.py`). Supporting Python modules remain organized under their own folders; there are no duplicate `server/app.py` or `server/app_v1.py` application entry points.

> **Important capability statement:** This is not a ChatGPT-equivalent system and is not a frontier model. The generative language model is a 377,568-parameter byte-level Transformer with a very small synthetic corpus. Open-ended generated text is often incoherent. The browser chat can answer covered educational questions by retrieving the exact authored Q&A entry from the local project corpus; this is retrieval, not proof that the neural model can reason. The image model is a small diffusion experiment trained on procedural 32×32 geometric images. Its current samples are noisy and it is not a useful general-purpose text-to-image model. Image understanding, speech, and video are not implemented or trained.

## Release contents

- `ai_core/`: independently implemented small decoder-only Transformer.
- `checkpoints/independent_lm_v0_1.pt`: v0.1 original random-init training checkpoint.
- `checkpoints/independent_lm_v0_2.pt`: v0.2 independently trained base checkpoint.
- `checkpoints/independent_lm_v0_4.pt`: continued v0.4 checkpoint.
- `checkpoints/independent_lm_v0_5.pt`: v0.5 instruction-tuned checkpoint used by the browser app.
- `checkpoints/image_diffusion_toy_v0_6.pt`: v0.6 tiny synthetic image diffusion experiment.
- `data/`: project-authored training examples and provenance manifest.
- `training/`: reproducible v0.2, v0.4 and v0.5 training scripts and records.
- `multimodal/image_generation/`: from-scratch toy conditional diffusion model and training script.
- `knowledge/`: transparent local lexical Q&A retrieval; no external embedding model.
- `memory/`: local SQLite conversation memory with search and delete support.
- `safety/`: basic request guardrails; not a complete safety system.
- `main.py`: the single integrated FastAPI backend entry point.
- `index.html`: the single responsive browser interface with chat and image-experiment tabs.
- `evaluation/`: loss reports and a sample sheet illustrating the image model's current limitations.
- `tests/`: automated tests for model artifacts, retrieval, memory, and package integrity.

## Milestones included

| Milestone | Included result | Actual status |
|---|---|---|
| v0.3 | Curated corpus expansion, JSONL instruction pairs, dataset provenance | Completed |
| v0.4 | Continue training the project's v0.2 model on v0.3 corpus | 400 steps completed |
| v0.5 | Answer-only supervised instruction tuning from v0.4 checkpoint | 180 steps completed; modest held-out loss improvement |
| v0.6 | Separate class-conditional diffusion experiment trained from random weights | 240 steps; synthetic data only; output quality remains noisy |
| v0.7 | Local SQLite conversation memory and lexical recall | Implemented and tested |
| v0.8 | Basic input validation and capability/evaluation disclosures | Implemented; not a full safety certification |
| v0.9 | Integrated local API and browser interface | Implemented and smoke-tested |
| v1.0 | Consolidated research-preview release, docs, checkpoints, tests | Packaged |

These milestone labels identify software and research releases. They do not mean every capability of a mature multimodal assistant has been achieved.

## Training provenance and independence

- The v0.1 base model was initialized from random weights.
- The v0.2 model was independently initialized from random weights; it did not load v0.1 weights.
- v0.4 continued from this project's own v0.2 weights.
- v0.5 instruction-tuned this project's own v0.4 weights.
- The v0.6 image model was initialized from random weights.
- No third-party pretrained model weights were loaded in these training runs.
- The training and inference code does not call external AI APIs.
- PyTorch, FastAPI, NumPy, and Pillow are general-purpose software libraries, not pretrained AI services.
- The included instructional corpus is small project-authored synthetic text drafted with AI writing assistance during development. That is text-data assistance, not imported model weights. It is not a broad web-scale dataset and must not be represented as one.
- The image training examples are generated procedurally by the included code; they are not photographs or a third-party image dataset.

The Transformer and diffusion families are established research approaches. This project does not claim that these underlying algorithm families are newly invented. Any claim that a particular architecture is novel requires prior-art research and controlled experiments.

## Measured training results

### v0.4 continued text-model training

- Model: 377,568 parameters; 3 Transformer blocks; hidden dimension 96; 4 heads; byte vocabulary 256; context 192 bytes.
- Starting checkpoint: this project's v0.2 weights.
- Additional optimizer steps: 400.
- Device: CPU.
- New v0.3 examples: 71; seven held out in the new split. The existing v0.2 held-out documents remained held out.
- Initial sampled validation next-byte loss: 2.23644.
- Final sampled validation next-byte loss: 2.23302.
- Best sampled validation loss: 2.22887.

### v0.5 supervised instruction tuning

- Started from `independent_lm_v0_4.pt`.
- Training examples: 64; validation examples: 7.
- Steps: 180.
- Loss was applied to answer bytes, not the user prompt bytes.
- Held-out answer loss: 2.36913 initially; 2.36323 at the end.
- This is a small improvement on seven held-out examples, not evidence of broad instruction-following or reasoning.

### v0.6 image experiment

- Model: 303,299 parameters.
- Random initialization; 240 optimizer steps.
- Dataset: 192 programmatically generated geometric images, six classes, 32×32 pixels.
- Training noise-prediction MSE sampled at step 60: 0.21850; step 120: 0.15420; step 180: 0.12610; step 240: 0.10284.
- The saved sample preview remains visibly noisy. The training loss does not establish useful image quality.

Full machine-readable records are in `training/training_history_v0_4.json`, `training/training_history_v0_5.json`, and `training/image_training_v0_6.json`.

## Install

Use Python 3.10+ (Python 3.11 or 3.12 recommended). A computer or cloud VM is recommended. The language model can run on CPU, but larger future training will need GPUs. Android/Pydroid may not support this PyTorch stack reliably.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch's installation build should match your hardware. The included research runs were performed on CPU. A CUDA GPU is optional for this small prototype.

## Run the integrated app

From the project root:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser on the same machine.

The app provides:

- **Chat:** first checks the project's small local Q&A collection for a close lexical match. Matching entries are labelled as local knowledge retrieval. If no match is found, the small v0.5 language model generates a response, labelled as experimental model output.
- **Memory:** if enabled, user messages and assistant replies are stored in `data/local_conversations.sqlite3` on the server machine. The UI includes a delete-memory control. This is local storage, not cloud sync.
- **Image experiment:** produces a sample using the toy diffusion checkpoint. The current results are noisy and limited to six geometric classes; do not expect arbitrary scenes or photorealism.

The app does not browse the web. The language model context is only 192 bytes, so prompts should be short. `GET /api/health` reports checkpoint and capability status. `GET /api/capabilities` reports what is not implemented.

**Security:** Keep the server bound to `127.0.0.1`. It has no production authentication, multi-user isolation, rate limiting, HTTPS termination, or hardened deployment configuration. Do not expose it publicly as-is. Conversation memory is stored unencrypted in a local SQLite database; do not enter secrets or sensitive personal information.

## Run models from the terminal

```bash
python inference/chat_v1.py
```

This calls only the local v0.5 checkpoint. Its open-ended output may be incoherent.

## Reproduce training

The released checkpoints are included, so normal use does not require retraining.

To continue a fresh v0.2 experiment through v0.4, preserve the supplied checkpoints first and run:

```bash
python training/train_v04.py --steps 400 --log-every 100
```

To train a new v0.5 instruction-tuning experiment, first create a v0.4 checkpoint, then run:

```bash
python training/train_v05_sft.py --steps 180 --batch-size 8
```

To reproduce the toy image experiment:

```bash
python -m multimodal.image_generation.train_toy_diffusion --steps 240 --samples 192 --batch-size 12
```

These commands overwrite their target checkpoint names. Copy the shipped checkpoints to a separate folder before conducting a new experiment.

## Evaluate and test

```bash
python -m pytest -q
```

The release includes automated tests for dataset records, checkpoint provenance, the image model state, lexical retrieval, and local memory. These tests validate the software pipeline; they do not establish model intelligence or deployment safety.

## Not yet achieved

- ChatGPT-level language quality or frontier-scale reasoning.
- Broad factual knowledge, multilingual competence, robust coding, or reliable mathematics.
- Useful arbitrary prompt-to-image generation or image understanding.
- Speech recognition, speech synthesis, audio understanding, video understanding, or video generation.
- Secure public multi-user deployment.
- A demonstrated novel neural-network architecture.

Those goals require much more data, GPU compute, research, testing, and engineering. The v1.0 label means the research-preview package is integrated and versioned, not that those open research goals are complete.

## Licence and data governance

Code is offered under MIT terms in `LICENSE`. Review dataset provenance and local legal requirements before adding third-party material. Training examples included here are provided for research and demonstration; they are not official academic, legal, medical, or financial guidance.
