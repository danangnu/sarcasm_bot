"""Render-only entry point; desktop app.py remains usable locally."""
from contextlib import asynccontextmanager
from html import escape
import os
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from starlette.concurrency import run_in_threadpool
from deploy.security import AccessBoundary, credentials

KIND = "starcasm"
TITLE = "StARCASM"
USERS = credentials()
if os.getenv('BOT_KIND') != KIND:
    raise RuntimeError('BOT_KIND does not match this repository.')
if not os.getenv('DEPLOY_MODEL_DIR') or not os.getenv('DATA_ROOT'):
    raise RuntimeError('Launch with python -m deploy.start to verify the model first.')
import app as legacy


def warmup():
    if KIND == 'starcasm':
        from core import ACTIVE_CLASSIFIER, analyze_text
        if ACTIVE_CLASSIFIER != 'transformer':
            raise RuntimeError('Transformer is required.')
        result = analyze_text('The meeting starts at nine today.')
        return result.score
    from core import classifier
    if classifier.allow_bootstrap:
        raise RuntimeError('Bootstrap models are disabled for deployment.')
    classifier.load()
    return classifier.analyze('The meeting starts at nine today.')['score']


@asynccontextmanager
async def lifespan(app):
    # A real tokenizer/model inference must succeed before the service accepts requests.
    await run_in_threadpool(warmup)
    yield


web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


@web.get('/readyz')
def ready():
    return {'status': 'ready', 'bot': KIND}


@web.get('/', response_class=HTMLResponse)
def portal():
    other = os.getenv('OTHER_BOT_URL', '').strip()
    valid = urlparse(other)
    other_name = 'HUMOR Bot' if KIND == 'starcasm' else 'StARCASM'
    other_link = (f'<a class="button secondary" href="{escape(other, quote=True)}">Open {other_name}</a>'
                  if valid.scheme == 'https' and valid.netloc and not valid.username else '<span>Companion bot link will appear after deployment.</span>')
    return """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>StARCASM &amp; HUMOR Bot | Research demonstration</title>
    <style>body{margin:0;background:#f4f7fb;color:#16283e;font:17px/1.6 system-ui,sans-serif}main{max-width:950px;margin:6vh auto;padding:28px}h1{font-size:clamp(32px,5vw,52px);line-height:1.15}h2{font-size:25px}small{letter-spacing:.12em;color:#315883}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:24px;margin:32px 0}article{background:white;border:1px solid #dbe3eb;border-radius:16px;padding:28px}.button{display:inline-block;padding:11px 20px;border-radius:8px;background:#214e77;color:white;text-decoration:none;margin-top:12px}.secondary{background:#276456}footer{font-size:14px;border-top:1px solid #ccd7e1;padding-top:18px}a:focus{outline:3px solid #e58c24;outline-offset:4px}</style>
    <main><small>CONVERSATIONAL RESEARCH TOOLS</small><h1>StARCASM &amp; HUMOR Bot</h1><p>Explore how two trained language models classify text and support short conversational exchanges.</p>
    <div class="grid"><article><h2>""" + TITLE + """</h2><p>Try text analysis, explore the model response, and submit corrections for review.</p><a class="button" href="/bot">Open """ + TITLE + """</a></article>
    <article><h2>""" + other_name + """</h2><p>Open the companion research demonstration.</p>""" + other_link + """</article></div>
    <h2>Before trying the demo</h2><p>A reviewer login is required. Use fictional examples and do not enter names, medical records, or other personal information. Submitted corrections are saved for review. Chat text may be sent to the configured external response provider; classification runs on the hosted model.</p>
    <p>These tools classify language. The scores do not measure cognitive ability, establish a diagnosis, or demonstrate rehabilitation benefit.</p><footer>Research demonstration · <a href="/admin" >Administrator sign-in</a></footer></main></html>"""


@web.get('/bot')
def bot():
    return FileResponse(Path(__file__).parent / 'static' / 'index.html')


@web.get('/admin')
def admin():
    return FileResponse(Path(__file__).parent / 'static' / 'admin.html')


@web.get('/health')
def health():
    # The UI needs status and thresholds, not internal paths or candidate history.
    data = legacy.health_check() if KIND == 'starcasm' else legacy.health()
    allowed = {'status','version','model_ready','tokenizer_ready','gemini_configured','gemini_model',
               'feedback_count','local_model_ready','model_loaded','model_source','bootstrap_allowed',
               'thresholds','threshold','active_classifier','model_version','max_input_chars'}
    result = {k:v for k,v in data.items() if k in allowed}
    result['retraining_available'] = False
    return result



@web.get('/admin/status')
def admin_status():
    data = legacy.admin_status()
    data['retraining_available'] = False
    data['hosted_read_only_model'] = True
    data['deployment_model_sha256'] = os.getenv('MODEL_ARCHIVE_SHA256')
    return data

web.mount('/', legacy.app)
app = AccessBoundary(web, USERS)
