## Sarcasm Detection Chatbot

Milestone 1 working demo for a sarcasm chatbot. The app accepts pasted text such as a tweet, headline, article paragraph, or custom message, scores it with the trained BiLSTM sarcasm model, and returns a chatbot response.

### What is included in Milestone 1

- FastAPI backend
- Static HTML/CSS/JavaScript frontend
- Saved BiLSTM model loading
- Saved tokenizer loading
- Sarcasm scoring endpoint
- Chat endpoint with Gemini support
- Local fallback response when Gemini is not configured
- Health endpoint for quick validation
- Improved demo UI for manual tweet/article text input

### Project structure

```text
.
├── app.py                    # FastAPI app and API routes
├── core.py                   # Model loading, scoring, Gemini/fallback response logic
├── models/
│   ├── bilstm_model.keras    # Trained sarcasm model
│   └── tokenizer.pkl         # Saved tokenizer
├── static/
│   ├── index.html            # Demo UI
│   ├── script.js             # Frontend behavior/API calls
│   └── style.css             # Demo styling
├── Training/                 # Original training assets
├── requirements.txt
└── .env
```

### Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Environment variables

Create or update `.env`:

```bash
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
GEMINI_MODEL=gemini-2.5-flash
SARCASM_THRESHOLD=0.5
REPLY_MIN_SCORE=0.65
MAX_RETRIES=3
```

`GEMINI_API_KEY` is optional for local UI testing. If it is missing, the backend uses a local fallback response so the demo flow can still be tested.

### Run

```bash
uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000/
```

You can also run:

```bash
python app.py
```

### API endpoints

Health check:

```bash
GET /health
```

Analyze text only:

```bash
POST /analyze
Content-Type: application/json

{
  "message": "Oh great, another Monday morning. Exactly what I needed."
}
```

Chat with bot:

```bash
POST /chat
Content-Type: application/json

{
  "message": "Oh great, another Monday morning. Exactly what I needed."
}
```

### Notes

- The current version analyzes manually entered or pasted text.
- It does not yet fetch full newspaper articles from URLs.
- It does not yet pull tweets directly from Twitter/X.
- Article URL extraction and Twitter/X integration should be handled as a later feature.

## Keras model compatibility note

If the app shows an error like:

```text
Unrecognized keyword arguments passed to Embedding: {'quantization_config': None}
```

that means the `.keras` model was saved with a different Keras/TensorFlow version than the runtime currently installed. Milestone 1 now includes a compatibility loader that automatically creates and loads:

```text
models/bilstm_model.compat.keras
```

The compatibility file only patches the model JSON configuration so older runtimes can deserialize it. The trained weights are not changed.

After updating this code, restart the FastAPI server and check:

```text
http://127.0.0.1:8000/health
```

`model_ready` and `tokenizer_ready` should be `true` before testing the chat screen.
