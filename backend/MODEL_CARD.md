# Model card — Independent AI Lab v1.0 Research Preview

## Intended use

Educational experimentation with small neural-network training, model checkpointing, local text generation, lexical Q&A retrieval, local conversation memory, and toy diffusion research.

## Not intended for

High-stakes decisions; medical, legal, financial, or safety advice; autonomous actions; public multi-user deployment; production customer support; reliable coding; general image generation; image understanding; speech or video tasks.

## Text model

- Architecture: compact decoder-only Transformer.
- Parameters: 377,568.
- Layers: 3.
- Hidden width: 96.
- Attention heads: 4.
- Context: 192 UTF-8 bytes.
- Vocabulary: 256 byte IDs.
- Initialization lineage: v0.2 was randomly initialized; v0.4 continued from v0.2; v0.5 was instruction-tuned from v0.4.
- v0.5 training: 180 answer-only supervised fine-tuning steps on 64 examples; seven separate examples for a small held-out answer-loss estimate.
- No third-party pretrained weights or external AI APIs were used by the training/inference pipeline.

## Text training data

The corpus contains a small project-authored educational seed set, 138 v0.2 documents, and 71 v0.3 documents. The v0.3 additions are included in the JSONL instruction set. Text was drafted with AI writing assistance for this development project. The corpus is synthetic, small, English-dominant, not representative of public language use, and not independently fact-checked as a general reference. Do not infer that the model has broad knowledge from its next-byte loss.

## Image model

- Architecture: compact class-conditional diffusion U-Net.
- Parameters: 303,299.
- Initialization: random.
- Training: 240 steps on 192 procedurally generated 32×32 images, across six geometric classes.
- No third-party image data or pretrained image weights used.
- Samples remain visibly noisy; capability is limited to a research experiment and must not be described as practical general text-to-image generation.

## Other components

- Local Q&A retrieval is lexical matching over included project-authored text. It is not a learned embedding system and not open-domain knowledge retrieval.
- Memory uses SQLite on the host machine. The database is unencrypted and can contain conversation text. The user interface exposes a clear-memory operation.
- Basic request checks do not constitute comprehensive safety or security measures.

## Limitations and risks

The neural language model can generate nonsensical or false text. It may reproduce training examples and has not undergone broad factuality, bias, privacy, memorization, adversarial, misuse, or safety evaluation. Its very short byte context limits prompt and response length. The synthetic image model's samples do not show reliable recognizable content. No image understanding, audio, or video model is included.

## Evaluation summary

- v0.4 sampled validation byte loss: 2.23644 initial; 2.23302 final; best observed 2.22887.
- v0.5 held-out answer-byte loss: 2.36913 initial; 2.36323 final on seven examples.
- v0.6 training noise MSE: 0.21850 at step 60; 0.10284 at step 240.
- These metrics are task-specific training diagnostics, not general capability scores.

## Deployment

Local research use only. Bind the API to loopback (`127.0.0.1`). The included app has no production authentication, rate limiting, encryption, or multi-user separation. Do not expose it to the public internet without substantial security work.
