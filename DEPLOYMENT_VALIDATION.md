# Deployment preparation validation

- Backend and hosted integration suite: 19 tests passed locally.
- Archive checks: file hashes, path allowlist, wrong bot kind, tampering, HTTPS requirement and failed download cleanup.
- Access checks: unauthenticated and reviewer/admin boundaries, encoded admin paths, cross-site writes and blocked hosted model mutations.
- Runtime checks: a tiny randomly initialized transformer completed actual local inference and fallback chat. This is a wiring test, not evaluation of the trained research model.
- Persistence: a synthetic correction remained available after starting the application in a separate process using the same persistent directory.
- Missing model weights: startup failed before readiness.
- Resource controls: two concurrent inference slots, excess request rejection, 64 KiB request limit.
- Render Blueprint: YAML parsed locally; account-side Blueprint validation and provisioning are pending.
- Browser JavaScript: syntax checked. Full browser visual acceptance is pending because the local browser installation was unavailable.

Test runtime: Python 3.12.14, CPU PyTorch 2.6.0, Transformers 4.55.0, FastAPI 0.115.12. Render is configured for Python 3.12.8; the first hosted build must verify the pinned runtime.

Not yet verified: supplied trained model inference in Render, actual peak memory/latency, external Gemini responses, Render restart/redeploy persistence, custom-domain origin behavior and browser acceptance. No Render resources were provisioned and no model weights or live secrets were committed.
