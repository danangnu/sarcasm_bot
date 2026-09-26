import asyncio
import base64
import hashlib
import json
import zipfile
from pathlib import Path
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from deploy.model_package import pack, extract, install, verify
from deploy.security import AccessBoundary, credentials

USERS = {'ADMIN_USER': 'admin', 'ADMIN_PASSWORD': 'admin-password-for-tests',
         'DEMO_USER': 'reviewer', 'DEMO_PASSWORD': 'review-password-for-tests'}


def auth(role='DEMO'):
    raw = (USERS[role+'_USER'] + ':' + USERS[role+'_PASSWORD']).encode()
    return {'Authorization': 'Basic ' + base64.b64encode(raw).decode()}


async def endpoint(request):
    if request.method == 'POST':
        return JSONResponse({'received': await request.json()})
    return JSONResponse({'ok': True})


@pytest.fixture
def client():
    wrapped = AccessBoundary(Starlette(routes=[Route('/{path:path}', endpoint, methods=['GET','POST','PUT','DELETE'])]), USERS)
    with TestClient(wrapped) as c:
        yield c


def test_anonymous_and_admin_boundary(client):
    assert client.get('/').status_code == 200
    assert client.get('/readyz').status_code == 200
    for path in ['/bot','/health','/admin','/admin.html','/admin/status','/static/admin.html','/docs','/openapi.json']:
        assert client.get(path).status_code == 401
    for path in ['/admin','/admin.html','/admin/status','/static/admin.html','/%61dmin/status','/static/x/../admin.html']:
        assert client.get(path,headers=auth()).status_code == 401
        assert client.get(path,headers=auth('ADMIN')).status_code == 200
    assert client.get('/bot',headers=auth()).status_code == 200
    assert client.get('/static/script.js',headers=auth()).headers['cache-control'] == 'no-store'


def test_cross_site_and_offline_actions(client):
    assert client.post('/feedback',headers={**auth(), 'Origin':'https://evil.example'},json={}).status_code == 403
    assert client.post('/feedback',headers={**auth(), 'Sec-Fetch-Site':'cross-site'},json={}).status_code == 403
    assert client.post('/feedback',headers=auth(),data={'x':'y'}).status_code == 415
    for path in ['/admin/retrain','/admin/promote','/admin/rollback']:
        assert client.post(path,headers=auth('ADMIN'),json={}).status_code == 403
    assert client.put('/admin/corrections/1',headers=auth('ADMIN'),json={}).status_code == 200
    assert client.put('/admin/corrections/1',headers=auth(),json={}).status_code == 401
    assert client.post('/feedback',headers={**auth(), 'Origin':'http://testserver'},json={'answer':'synthetic'}).json()['received']['answer']=='synthetic'


def test_request_size_and_invalid_auth(client):
    assert client.post('/feedback',headers=auth(),json={'x':'x'*65536}).status_code==413
    for value in ['Basic !!!','Bearer xyz','Basic Zm9v','Basic /w==']:
        assert client.get('/health',headers={'Authorization':value}).status_code==401


def test_concurrency_limit():
    import httpx
    async def run():
        entered=0
        both=asyncio.Event()
        release=asyncio.Event()
        async def slow(request):
            nonlocal entered
            entered+=1
            if entered==2: both.set()
            await release.wait()
            return JSONResponse({'ok':True})
        app=AccessBoundary(Starlette(routes=[Route('/analyze',slow,methods=['POST'])]),USERS)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as c:
            a=asyncio.create_task(c.post('/analyze',headers=auth(),json={}))
            b=asyncio.create_task(c.post('/analyze',headers=auth(),json={}))
            await asyncio.wait_for(both.wait(),2)
            third=await c.post('/analyze',headers=auth(),json={})
            assert third.status_code==429
            release.set()
            assert (await a).status_code==200 and (await b).status_code==200
            assert (await c.post('/analyze',headers=auth(),json={})).status_code==200
    asyncio.run(run())


def model_dir(tmp,kind):
    tmp.mkdir(parents=True,exist_ok=True)
    files={'config.json':{'model_type':'roberta'},'tokenizer_config.json':{},'tokenizer.json':{},
           'model_metadata.json':{'threshold':0.2},'thresholds.json':{'low_threshold':0.2,'high_threshold':0.8}}
    for n,v in files.items(): (tmp/n).write_text(json.dumps(v))
    (tmp/'model.safetensors').write_bytes(b'test fixture only; not real inference weights')
    return tmp


@pytest.mark.parametrize('kind',['starcasm','humor'])
def test_package_roundtrip_and_tamper(tmp_path,kind):
    source=model_dir(tmp_path/'source',kind)
    (source/'.env').write_text('EXAMPLE=do not include')
    (source/'corrections.csv').write_text('do not include')
    output=tmp_path/'package.zip'
    pack(source,output,kind)
    dest=tmp_path/'dest';dest.mkdir()
    extract(output,dest,kind)
    assert not (dest/'.env').exists() and not (dest/'corrections.csv').exists()
    assert output.with_suffix('.zip.sha256').read_text().strip()==hashlib.sha256(output.read_bytes()).hexdigest()
    (dest/'model.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError): verify(dest,kind)


@pytest.mark.parametrize('name',['../escape','/absolute','nested/config.json','evil.py'])
def test_archive_rejects_paths(tmp_path,name):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr(name,'x')
    with pytest.raises(ValueError):extract(archive,tmp_path/'out','starcasm')
    assert not (tmp_path/'escape').exists()


def test_cached_archive_integrity_and_kind(tmp_path):
    model=model_dir(tmp_path/'source','starcasm');archive=tmp_path/'model.zip';pack(model,archive,'starcasm')
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    cache=tmp_path/'runtime'/'model-cache';cache.mkdir(parents=True)
    (cache/(digest+'.zip')).write_bytes(archive.read_bytes())
    dest=install(tmp_path/'runtime','https://unused.example/model.zip',digest,'starcasm')
    (dest/'config.json').write_text('tampered')
    # Restart reconstructs the model from the checksum-pinned archive.
    dest=install(tmp_path/'runtime','https://unused.example/model.zip',digest,'starcasm')
    assert json.loads((dest/'config.json').read_text())['model_type']=='roberta'
    with pytest.raises(ValueError):install(tmp_path/'runtime','https://unused.example/model.zip',digest,'humor')
    with pytest.raises(ValueError):install(tmp_path,'http://insecure.example/model.zip',digest,'starcasm')


def test_failed_download_leaves_no_model(tmp_path,monkeypatch):
    import io
    class Opener:
        def open(self,*a,**k): return io.BytesIO(b'wrong content')
    monkeypatch.setattr('urllib.request.build_opener',lambda *a:Opener())
    with pytest.raises(RuntimeError):install(tmp_path,'https://example.test/model.zip','a'*64,'starcasm')
    assert not list((tmp_path/'model-cache').glob('*.zip'))
    assert not list((tmp_path/'model-cache').glob('*.download'))


def test_weak_credentials_rejected(monkeypatch):
    for k,v in USERS.items():monkeypatch.setenv(k,v)
    assert credentials()==USERS
    monkeypatch.setenv('ADMIN_PASSWORD','short')
    with pytest.raises(RuntimeError):credentials()
