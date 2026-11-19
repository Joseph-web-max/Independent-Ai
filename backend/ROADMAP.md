# Independent AI Lab — v0.3 to v1.0 release record

## v0.3 — data pipeline and corpus expansion — completed

- Added 71 project-authored educational Q&A documents and JSONL instruction pairs.
- Added a provenance manifest and documented synthetic-data limitations.
- Kept the UTF-8 byte tokenizer for compatibility with the existing model. A tokenizer replacement would require a deliberate vocabulary/embedding migration and separate comparison; it is not silently substituted.

## v0.4 — continued base-model training — completed

- Continued the project's own v0.2 checkpoint; no third-party weights or AI APIs.
- Added 400 optimizer steps on the v0.3 additions and existing training corpus.
- Preserved held-out data splits.
- Measured validation next-byte loss from 2.23644 to 2.23302 (best sampled 2.22887).

## v0.5 — supervised instruction tuning — completed as a small experiment

- Fine-tuned v0.4 for 180 steps.
- Applied training loss only to answer bytes.
- Used 64 training and 7 held-out project-authored examples.
- Held-out answer loss changed from 2.36913 to 2.36323. The result is modest and the model remains unreliable.

## v0.6 — image-generation research experiment — completed as a toy proof of pipeline

- Added a 303,299-parameter class-conditional diffusion U-Net.
- Started from random weights and trained 240 steps on 192 procedurally generated 32×32 geometric images.
- The saved model can sample conditioned outputs, but the preview is visibly noisy. It is not general text-to-image capability.

## v0.7 — local memory — implemented

- SQLite storage for conversation turns.
- Lightweight lexical search and recent-turn retrieval.
- User-facing clear-memory endpoint and UI control.
- Storage is local and unencrypted; production privacy and security work remains.

## v0.8 — basic guardrails and evaluation disclosure — implemented

- Request length validation and private-key pattern rejection.
- Explicit capability endpoint and limitations in the UI.
- Existing tests validate code paths and provenance metadata.
- This is not a complete policy, red-team, privacy, or security program.

## v0.9 — integrated local application — implemented

- FastAPI backend loads only project checkpoints.
- Browser app supports local chat, local memory and toy image diffusion sampling.
- Smoke tests were performed against health, chat retrieval and image endpoints.
- No external AI API or cloud model dependency is present.

## v1.0 — consolidated research preview — packaged

- One source tree, checkpoint set, training scripts, reports, documentation, and tests.
- All stages can be downloaded together.
- Frontier performance, useful image quality, image understanding, audio and video remain future research tasks.

## Future research beyond v1.0

1. Create a much larger, legally reviewed, deduplicated multilingual corpus with documented licences and consent basis.
2. Train a subword tokenizer from project-owned data and compare it with the byte baseline.
3. Scale the language model with a documented GPU budget, larger context, reproducible checkpoints, and leakage-resistant evaluations.
4. Expand high-quality instruction data and evaluate on unseen reasoning, math, coding, factuality and multilingual tasks.
5. Develop and train separate image understanding, image generation, audio and video models from scratch using suitable datasets and substantial compute.
6. Add privacy controls, authentication, security hardening, monitoring, independent safety testing, and deployment operations.
7. Conduct prior-art review before asserting that any neural architecture is novel.
