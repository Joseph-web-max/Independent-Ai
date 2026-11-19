# Start here — Independent AI Lab v1.0

This ZIP contains **one browser frontend (`index.html`) and one application backend (`main.py`)**. Supporting model, tokenizer, memory, knowledge, training, evaluation, and test modules live in their own folders; they are not duplicate app entry points.

## 1. Install

1. Extract the ZIP archive.
2. Install Python 3.10+ (Python 3.11 or 3.12 recommended).
3. Open a terminal in the extracted project folder (the folder containing `main.py`).
4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

## 2. Run the app

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` in a browser on the same computer.

Alternatively, use `scripts/run_local.sh` (Linux/macOS) or `scripts/run_local_windows.bat` (Windows).

## Important limitations

The included language model is a small experimental research model and may produce incoherent or incorrect answers. Covered questions may be answered by a clearly labelled local knowledge-retrieval component. The image model is a tiny synthetic diffusion experiment and produces noisy geometric samples, not polished general-purpose images. Image understanding, speech, and video are not implemented.

Keep the server bound to `127.0.0.1`. The app is not hardened for public multi-user deployment and has no production authentication or rate limiting. Conversation memory is stored in a local SQLite file.

Read `README.md`, `MODEL_CARD.md`, and `TRAINING_RESULTS.md` for complete details. The supplied checkpoints mean you can launch the app without retraining.
