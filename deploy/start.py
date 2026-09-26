"""Render entry point: verify model before starting exactly one web worker."""
import os
from pathlib import Path
import sys
from deploy.model_package import install
from deploy.security import credentials


def main():
    credentials()
    root = Path(os.environ.get('DATA_ROOT', '/var/data')).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model = install(root, os.environ.get('MODEL_ARCHIVE_URL', ''),
                    os.environ.get('MODEL_ARCHIVE_SHA256', ''), os.environ['BOT_KIND'])
    os.environ['DEPLOY_MODEL_DIR'] = str(model)
    os.environ['DATA_ROOT'] = str(root)
    os.environ['REQUIRE_TRANSFORMER'] = 'true'
    os.environ['ALLOW_BOOTSTRAP_MODEL'] = 'false'
    os.environ['HUMOR_MODEL_PATH'] = str(model)
    os.environ['HUMOR_RUNTIME_ROOT'] = str(root)
    os.environ['HUMOR_FEEDBACK_DB'] = str(root / 'feedback' / 'humor_feedback.db')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.execv(sys.executable, [sys.executable, '-m', 'uvicorn', 'hosted:app', '--host', '0.0.0.0',
                             '--port', os.environ.get('PORT', '8000'), '--workers', '1', '--no-access-log'])

if __name__ == '__main__':
    main()
