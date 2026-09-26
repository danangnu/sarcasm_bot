# Render deployment — StARCASM

This branch prepares an invitation-only research demonstration. It does not provision Render resources or contain trained model weights. The hosted entry point is `python -m deploy.start`; `uvicorn app:app` is the local desktop/development entry point and does not apply the hosted access boundary.

## 1. Package the actual evaluated model locally

Use the active model folder, including matching tokenizer files and thresholds. The packer uses only Python's standard library and excludes datasets, corrections, API keys, candidate history and training checkpoints. It accepts a single `model.safetensors` file (no pickle weights or sharded checkpoints).

```powershell
python -m deploy.model_package --kind starcasm --model-dir models/transformer_model --output starcasm-model.zip
```

Keep the printed `MODEL_ARCHIVE_SHA256` and the adjacent `.sha256` file. The model package must have `config.json`, `tokenizer_config.json`, `tokenizer.json` (or `vocab.json` plus `merges.txt`), `model.safetensors`, and `model_metadata.json with a numeric threshold`. Review that the folder is the intended active release before packaging: the checksum verifies identity/integrity, not model quality.

Upload the ZIP to a project-controlled object store. Use an HTTPS download URL, including a signed URL if the object is private. Put the URL in Render's environment settings, never in Git. Keep it valid for later cold starts and disaster recovery. No storage account is provisioned by this PR.

## 2. Create the Render service

Connect `danangnu/sarcasm_bot` to Render. Create a Blueprint from this deployment branch (or merge first and use main). Review the proposed paid 2 CPU / 4 GB service and 10 GB disk before creating resources; this is initial demo sizing, not a load-tested capacity claim.

Supply `MODEL_ARCHIVE_URL`, `MODEL_ARCHIVE_SHA256`, and different strong `DEMO_PASSWORD` and `ADMIN_PASSWORD` values (at least 16 characters). Reviewer and administrator usernames are set in the Blueprint and can be changed. Do not reuse study participant credentials. The two apps have separate credentials and storage.

The startup process downloads at runtime, verifies the archive hash and every model file, loads the local model and runs a real inference before `/readyz` becomes available. Missing or incorrect model files stop startup. StARCASM cannot fall back to BiLSTM; HUMOR cannot bootstrap another model. Only one worker/instance is supported with this storage design. CPU-only inference dependencies omit TensorFlow and training libraries. The Blueprint pins Python 3.12.8; local checks used Python 3.12.14. Confirm the pinned runtime during the first Render build.

`/` is the public shared landing page. `/bot` requires a reviewer or administrator login. `/admin` and all admin APIs require administrator credentials. HTTP Basic authentication is for the HTTPS preview, not the future participant account system. Use a private browser window to switch accounts or sign out by closing it. Hosted training, promotion and rollback return 403; update the verified package and redeploy to change the model. Corrections can still be reviewed and edited online.

## 3. Chat responses and companion link

Local fallback replies work without external credentials. To enable generated replies, add `GEMINI_API_KEY` and a supported `GEMINI_MODEL` in Render settings after checking availability in the project's provider account. Never put the key in browser code. The landing page discloses external chat processing.

After both services are live, set `OTHER_BOT_URL` to the companion HTTPS service URL on each service and restart. `RENDER_EXTERNAL_URL` supplies the expected browser origin; if using a custom domain, set `PUBLIC_ORIGIN` to that exact HTTPS origin.

## 4. Acceptance checks after deployment

1. `/readyz` returns 200 only after model loading and warm-up succeed. Missing/corrupt weights must fail deployment; never replace them with a dummy model.
2. `/` shows the landing page; both bot links work. `/bot` prompts for a login. Check analysis and chat with a few fictional examples.
3. Anonymous `/health`, `/admin`, `/admin.html`, `/static/admin.html`, and admin API access returns 401. Reviewer credentials cannot read or modify admin records. Administrator login opens the correction review screen.
4. Save one clearly synthetic correction; reload, restart, then redeploy the same release. Confirm it remains in the review screen. The disk is mounted at `/var/data`; source-tree feedback/database files are not imported.
5. Check that cross-site writes return 403, model-changing endpoints return 403 even for administrators, and the third simultaneous inference receives 429 when both processing slots are occupied.
6. Review the loaded model version, thresholds and known evaluation examples. Compare outcomes to the local verified model. Record actual latency and memory before inviting more reviewers.
7. Verify Gemini and fallback behavior separately. Classification success does not establish external-provider availability.

## Storage, backups and release operation

Feedback and application state live under `/var/data`; immutable packages and extracted models live under `/var/data/model-cache`. Disk-backed services are single instance and incur restart downtime on deployment. Plan downtime with reviewers. Old model packages remain for rollback and consume disk space; remove only unused package hashes during a maintenance window.

Export StARCASM corrections through its authenticated export endpoint. For HUMOR, use SQLite's backup API or `.backup` into a consistent backup file before copying it off the disk; do not copy a live database without its WAL. Keep project-controlled off-service backups and test restoration. Disk snapshots are not a replacement for consistent SQLite backups. Retention and clinical collection are separate study decisions; this release is for fictional review examples.

For rollback, restore the prior code release and prior package URL/SHA, then restart and repeat acceptance checks. Do not use the desktop model-promotion controls against the deployed package.

## Automated checks

```bash
pip install -r requirements-render.txt pytest httpx pandas scikit-learn
python -m pytest tests -q
```

Deployment tests cover archive integrity/path rejection, access controls, CSRF, request bounds, concurrency, and storage. Small generated test models can validate runtime wiring but cannot establish the quality or availability of the actual trained weights. Live Render verification remains required.

Render references: https://render.com/docs/blueprint-spec and https://render.com/docs/disks (checked 2026-09-26).
