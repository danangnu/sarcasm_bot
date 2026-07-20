# Milestone 2 Final Validation Checklist

## Automated checks

```powershell
python -m unittest discover -s tests -v
```

## Browser checks

1. Open `http://127.0.0.1:8000/health`.
2. Confirm `model_ready` and `tokenizer_ready` are `true`.
3. Open `http://127.0.0.1:8000/`.
4. Test a sarcastic sentence and confirm scores from 50–64% show **Moderate**, 65–79% show **High**, and 80–100% show **Very high**.
5. Test a sincere sentence and confirm the same confidence bands are applied symmetrically to the predicted non-sarcastic class.
6. Confirm Gemini responses show the **Gemini** badge when quota is available.
7. Confirm provider failures show a short **Local fallback** message without raw API diagnostics.
8. Verify Analyze only, Send to bot, Clear, Copy, theme toggle, character count, Enter, and Shift+Enter.
9. Resize the browser to mobile width and confirm no horizontal overflow.

## Suggested demo inputs

- Sarcastic: `Amazing. The application crashed for the third time today.`
- Sarcastic: `Oh great, another Monday morning. Exactly what I needed.`
- Sincere: `I really appreciate your help with the project today.`
- Neutral: `The meeting starts at 10:00 tomorrow morning.`

## Interpretation note

The confidence label describes how far the selected class is from the 50% decision boundary. It is not proof of intent and is not a measured real-world accuracy score.
