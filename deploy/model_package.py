"""Package and install flat, safetensors-only model archives; no training data."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import urllib.request
import zipfile

MAX_BYTES = 2 * 1024**3
ALLOWED = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
           "vocab.json", "vocab.txt", "merges.txt", "added_tokens.json", "model.safetensors",
           "model_metadata.json", "thresholds.json"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_model(folder, kind):
    required = {'config.json', 'tokenizer_config.json', 'model.safetensors'}
    required.add('model_metadata.json' if kind == 'starcasm' else 'thresholds.json')
    if not all((folder / name).is_file() for name in required):
        raise ValueError('Model package lacks required weights, configuration or thresholds.')
    if not (folder / 'tokenizer.json').is_file():
        if not ((folder / 'vocab.json').is_file() and (folder / 'merges.txt').is_file()):
            raise ValueError('Tokenizer files are missing.')
    if not (folder / 'model.safetensors').stat().st_size:
        raise ValueError('Weights file is empty.')
    config = json.loads((folder / 'config.json').read_text())
    if config.get('auto_map'):
        raise ValueError('Remote model code is not allowed.')
    metadata = json.loads((folder / ('model_metadata.json' if kind == 'starcasm' else 'thresholds.json')).read_text())
    values = [metadata.get('threshold')] if kind == 'starcasm' else [metadata.get('low_threshold'), metadata.get('high_threshold')]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Explicit finite thresholds between 0 and 1 are required.')
    if kind == 'humor' and values[0] >= values[1]:
        raise ValueError('Humor thresholds must be ordered.')


def pack(folder, output, kind):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    validate_model(folder, kind)
    files = sorted(p for p in folder.iterdir() if p.name in ALLOWED and p.is_file())
    if any(p.is_symlink() for p in files) or sum(p.stat().st_size for p in files) > MAX_BYTES:
        raise ValueError('Unsafe or oversized model package.')
    manifest = {'format': 1, 'kind': kind, 'files': {p.name: sha256(p) for p in files}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as z:
        for p in files:
            z.write(p, p.name)
        z.writestr('manifest.json', json.dumps(manifest, indent=2))
    digest = sha256(output)
    output.with_suffix(output.suffix + '.sha256').write_text(digest + '\n')
    print(f'Package: {output}\nMODEL_ARCHIVE_SHA256={digest}')


def verify(folder, kind):
    manifest = json.loads((folder / 'manifest.json').read_text())
    files = manifest.get('files', {})
    if manifest.get('format') != 1 or manifest.get('kind') != kind or not files:
        raise ValueError('Model manifest does not match this bot.')
    if not set(files) <= ALLOWED:
        raise ValueError('Unexpected model file.')
    if {p.name for p in folder.iterdir()} != set(files) | {'manifest.json'}:
        raise ValueError('Model directory differs from manifest.')
    for name, digest in files.items():
        p = folder / name
        if p.is_symlink() or not p.is_file() or sha256(p) != digest:
            raise ValueError('Model checksum verification failed.')
    validate_model(folder, kind)


def extract(archive, destination, kind):
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        names = [i.filename for i in entries]
        if len(names) != len(set(names)) or not set(names) <= ALLOWED | {'manifest.json'}:
            raise ValueError('Archive must contain only allowed flat model files.')
        if sum(i.file_size for i in entries) > MAX_BYTES:
            raise ValueError('Model expands beyond the size limit.')
        for i in entries:
            if i.is_dir() or stat.S_ISLNK(i.external_attr >> 16):
                raise ValueError('Directories and symbolic links are not allowed.')
        z.extractall(destination)  # filenames were checked against a flat allowlist
    verify(destination, kind)


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith('https://'):
            raise ValueError('Model redirects must use HTTPS.')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def install(root, url, expected, kind):
    if not re.fullmatch(r'[a-fA-F0-9]{64}', expected or ''):
        raise ValueError('MODEL_ARCHIVE_SHA256 must be the package SHA-256.')
    if not url.startswith('https://'):
        raise ValueError('MODEL_ARCHIVE_URL must use HTTPS.')
    expected = expected.lower()
    root = Path(root)
    cache = root / 'model-cache'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / (expected + '.zip')
    if not archive.exists() or sha256(archive) != expected:
        part = cache / (expected + '.download')
        try:
            opener = urllib.request.build_opener(HTTPSRedirect())
            with opener.open(url, timeout=60) as response, part.open('wb') as out:
                total = 0
                while chunk := response.read(1024**2):
                    total += len(chunk)
                    if total > MAX_BYTES + 1024**2:
                        raise ValueError('Download exceeds model package limit.')
                    out.write(chunk)
            if sha256(part) != expected:
                raise ValueError('Downloaded archive checksum does not match.')
            part.replace(archive)
        except Exception:
            part.unlink(missing_ok=True)
            raise RuntimeError('Model download failed; check the URL, expiry, size and SHA-256.') from None
    # Always re-extract the verified archive, so modified cached metadata cannot be trusted.
    target = cache / expected
    with tempfile.TemporaryDirectory(dir=cache, prefix='install-') as tmp:
        staging = Path(tmp) / 'model'
        staging.mkdir()
        extract(archive, staging, kind)
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    return target


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['starcasm', 'humor'], required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    pack(a.model_dir, a.output, a.kind)
