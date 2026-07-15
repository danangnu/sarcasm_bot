# Sarcasm Detection Chatbot — Milestone 2

Milestone 2 builds on the working FastAPI + BiLSTM + Gemini demo and improves reliability, diagnostics, and presentation.

## Included in this development pass

- Gemini timeout and retry handling
- Clear fallback reason and response-source metadata
- Authentication/quota/request error classification
- User and bot confidence labels
- Plain-language model interpretation note
- Character counter and 8,000-character input limit
- Loading and disabled-button states
- Timestamps and copy buttons
- Clear-chat action
- Light/dark theme toggle
- Improved mobile layout and error messages
- Existing Keras compatibility loader retained

## Setup

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
notepad .env
```

Add your Gemini key to `.env`:

```env
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-2.5-flash
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

## API changes

`POST /analyze` now also returns:

- `confidence`
- `explanation`
- `character_count`

`POST /chat` now also returns:

- `user_confidence`
- `reply_confidence`
- `user_explanation`
- `response_source`
- `fallback_reason`
- `generation_attempts`

## Scope note

This version analyzes manually entered or pasted text. Automatic article URL extraction and direct Twitter/X fetching remain separate features.
