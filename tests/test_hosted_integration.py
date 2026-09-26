"""Hosted integration uses tiny random weights solely to verify wiring, never model quality."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

KIND = 'starcasm'


@pytest.fixture
def fixture_model(tmp_path):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    folder=tmp_path/'model';folder.mkdir()
    (folder/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nthe\nmeeting\nstarts\nat\nnine\ntoday\n.\n')
    tokenizer=BertTokenizerFast(vocab_file=str(folder/'vocab.txt'))
    tokenizer.save_pretrained(folder)
    cfg=BertConfig(vocab_size=12,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,num_labels=2)
    BertForSequenceClassification(cfg).save_pretrained(folder,safe_serialization=True)
    (folder/'model_metadata.json').write_text(json.dumps({'threshold':0.2,'model_version':'TEST-ONLY','max_length':32}))
    (folder/'thresholds.json').write_text(json.dumps({'low_threshold':0.2,'high_threshold':0.8}))
    return folder


def execute(model, data, phase):
    env=os.environ.copy()
    env.update(DEPLOY_MODEL_DIR=str(model), DATA_ROOT=str(data), BOT_KIND=KIND, REQUIRE_TRANSFORMER='true',
               HUMOR_MODEL_PATH=str(model), HUMOR_RUNTIME_ROOT=str(data), HUMOR_FEEDBACK_DB=str(data/'feedback'/'humor_feedback.db'),
               ALLOW_BOOTSTRAP_MODEL='false', HF_HUB_OFFLINE='1', GEMINI_API_KEY='',
               DEMO_USER='reviewer', DEMO_PASSWORD='reviewer-test-password',
               ADMIN_USER='admin', ADMIN_PASSWORD='administrator-test-password', TEST_PHASE=phase)
    script=r"""
import os
from starlette.testclient import TestClient
from hosted import app
kind=os.environ['BOT_KIND']
phase=os.environ['TEST_PHASE']
with TestClient(app) as c:
    assert c.get('/readyz').json()['status']=='ready'
    assert 'StARCASM' in c.get('/').text
    assert c.get('/bot').status_code==401
    reviewer=('reviewer','reviewer-test-password')
    admin=('admin','administrator-test-password')
    assert c.get('/bot',auth=reviewer).status_code==200
    assert c.get('/style.css' if kind=='starcasm' else '/static/style.css',auth=reviewer).status_code==200
    assert c.get('/admin/status',auth=reviewer).status_code==401
    assert c.get('/admin/status',auth=admin).json()['hosted_read_only_model'] is True
    h=c.get('/health',auth=reviewer).json()
    assert h['retraining_available'] is False
    assert 'runtime_root' not in h and 'active_model' not in h
    if kind=='starcasm':
        assert h['active_classifier']=='transformer' and h['model_ready']
        payload={'message':'The meeting starts at nine today.'}
        correction={'text':'Synthetic review example','predicted_label':'Sarcastic','predicted_score':0.9,'correct_label':'not_sarcastic','source':'analyze'}
    else:
        assert h['model_loaded'] and not h['bootstrap_allowed']
        payload={'text':'The meeting starts at nine today.'}
        correction={'input_text':'Synthetic review example','original_label':'Humorous','original_score':0.9,'corrected_label':'Not Humorous','source':'analyze','model_source':'TEST-ONLY'}
    assert c.post('/analyze',auth=reviewer,json=payload).status_code==200
    chat=c.post('/chat',auth=reviewer,json={'message':'The meeting starts at nine today.'})
    assert chat.status_code==200 and chat.json()['response_source']=='Local fallback'
    if phase=='write':
        assert h['feedback_count']==0
        assert c.post('/feedback',auth=reviewer,json=correction).status_code==200
    assert c.get('/health',auth=reviewer).json()['feedback_count']==1
    assert c.post('/admin/retrain',auth=admin,json={}).status_code==403
print('PASS hosted '+phase)
"""
    return subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).parents[1],env=env,text=True,capture_output=True,timeout=60)


def test_hosted_inference_and_restart(fixture_model,tmp_path):
    for phase in ['write','restart']:
        r=execute(fixture_model,tmp_path/'data',phase)
        assert r.returncode==0, r.stdout+r.stderr


def test_missing_weights_cannot_be_ready(fixture_model,tmp_path):
    (fixture_model/'model.safetensors').unlink()
    r=execute(fixture_model,tmp_path/'data','write')
    assert r.returncode!=0
    assert 'PASS hosted' not in r.stdout
