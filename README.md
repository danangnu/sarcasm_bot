# StARCASM

The current source includes transformer inference. For the hosted demonstration, follow [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md). The trained model is distributed separately; the repository alone is not a complete deployment.

The local/desktop instructions below describe earlier delivery workflows.

# Sarcasm Detection Chatbot — Milestone 2 Pass 3

This build continues the working FastAPI + BiLSTM + Gemini demo and focuses on safe error handling, clearer results, and client-ready presentation.

## Pass 3 improvements

- Confidence labels are now calibrated from the probability of the predicted class.
- The scale is symmetric for sarcastic and non-sarcastic predictions.
- Confidence bands are **50–64% Moderate**, **65–79% High**, and **80–100% Very high**.
- The UI includes a visible confidence-scale legend.
- Boundary tests verify the confidence bands on both sides of the 0.50 threshold.

- Raw Gemini provider errors are no longer returned to the browser.
- Friendly fallback messages are mapped from structured error codes.
- Quota and authentication errors stop immediately instead of retrying unnecessarily.
- Detailed Gemini diagnostics are written to `logs/sarcasm_bot.log`.
- Result cards separate the user's score, bot score, explanation, and response source.
- Source badges show **Gemini**, **Local fallback**, or **Analyze only**.
- Gemini runtime status changes after a real request.
- API errors are shown as short user-facing messages.
- The existing Keras compatibility loader remains enabled.

## Security

The real `.env` file is intentionally excluded from this package and from Git. Create it locally from `.env.example`.

If an API key was ever committed to GitHub or shared publicly, revoke it and create a new key.

## Setup

```powershell
cd C:\path\to\sarcasm_bot
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
notepad .env
```

Add your Gemini configuration:

```env
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-3.6-flash
GEMINI_TIMEOUT_SECONDS=20
MAX_RETRIES=3
```

Run:

```powershell
uvicorn app:app --reload
```

Open:

- UI: `http://127.0.0.1:8000/`
- Health: `http://127.0.0.1:8000/health`
- API docs: `http://127.0.0.1:8000/docs`

## Chat response metadata

`POST /chat` returns:

- `response_source`
- `fallback_code`
- `fallback_message`
- `generation_attempts` (diagnostic API field; hidden from the UI)
- `request_duration_ms` (diagnostic API field; hidden from the UI)

No raw Gemini exception text is returned to the browser.

## Validation

Run the lightweight automated tests:

```powershell
python -m unittest discover -s tests -v
```

The helper tests cover text cleaning, calibrated confidence boundaries, predicted-class probability, and safe Gemini fallback messages. Full model and Gemini validation still require the local TensorFlow environment and API configuration.

See `FINAL_VALIDATION.md` for the client-demo test checklist.

## Scope

This version analyzes manually entered or pasted text. Automatic article URL extraction and direct Twitter/X fetching remain separate features.

## Delivery feedback workflow

Each analysis result includes **Correct** and **Incorrect** controls. Incorrect classifications are saved to `feedback/corrections.csv` beside the application. Corrections do not immediately change the active model.

In the source/development build, **Retrain candidate model** starts `training/train_from_feedback.py`. It creates a new candidate under `models/headline_candidate_v3` and validates it without replacing the active model. Review the reports before promotion.

The client executable intentionally disables local retraining because a packaged executable does not contain a complete editable Python training environment. It still records corrections, which can be returned to the development team for controlled retraining.

## Build the Windows executable

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install pyinstaller
.\build_exe.ps1
```

Place the client's `.env` beside `dist\SarcasmBot\SarcasmBot.exe`.
