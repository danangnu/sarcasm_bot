"""Access boundary for the single-worker, invitation-only hosted demonstration."""
import asyncio
import base64
import binascii
import hmac
import os
import posixpath
from starlette.responses import JSONResponse


def credentials():
    values = {k: os.getenv(k, '') for k in ('DEMO_USER', 'DEMO_PASSWORD', 'ADMIN_USER', 'ADMIN_PASSWORD')}
    if any(not v for v in values.values()) or any(len(values[k]) < 16 for k in ('DEMO_PASSWORD', 'ADMIN_PASSWORD')):
        raise RuntimeError('Set separate demo/admin usernames and passwords of at least 16 characters.')
    if values['DEMO_USER'] == values['ADMIN_USER'] or values['DEMO_PASSWORD'] == values['ADMIN_PASSWORD']:
        raise RuntimeError('Demo and administrator credentials must differ.')
    return values


class AccessBoundary:
    def __init__(self, app, users):
        self.app = app
        self.users = users
        self.slots = asyncio.Semaphore(2)

    def role(self, authorization):
        try:
            scheme, token = authorization.split(' ', 1)
            if scheme.lower() != 'basic':
                return None
            user, password = base64.b64decode(token, validate=True).decode('utf-8').split(':', 1)
        except (ValueError, UnicodeError, binascii.Error):
            return None
        for role in ('ADMIN', 'DEMO'):
            user_ok = hmac.compare_digest(user.encode(), self.users[role + '_USER'].encode())
            pass_ok = hmac.compare_digest(password.encode(), self.users[role + '_PASSWORD'].encode())
            if user_ok & pass_ok:
                return role
        return None

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        raw_path, method = scope['path'], scope['method']
        path = '/' + posixpath.normpath(raw_path).lstrip('/')
        headers = {k.decode().lower(): v.decode() for k, v in scope['headers']}
        async def respond(code, detail, extra=None):
            response = JSONResponse({'detail': detail}, status_code=code, headers=extra)
            await response(scope, receive, safe_send)
        async def safe_send(message):
            if message['type'] == 'http.response.start':
                message['headers'] = list(message.get('headers', [])) + [
                    (b'cache-control', b'no-store'), (b'x-content-type-options', b'nosniff'),
                    (b'x-frame-options', b'DENY'), (b'referrer-policy', b'no-referrer')]
            await send(message)
        if raw_path in ('/', '/readyz') and method in ('GET', 'HEAD'):
            return await self.app(scope, receive, safe_send)
        admin = (path.startswith('/admin') or path.startswith('/static/admin')
                 or path in ('/docs', '/redoc', '/openapi.json'))
        role = self.role(headers.get('authorization', ''))
        if role is None or (admin and role != 'ADMIN'):
            return await respond(401, 'Sign in with the appropriate study demonstration account.',
                                 {'WWW-Authenticate': 'Basic realm="Research demo", charset="UTF-8"'})
        # The web process does not train or change the active model. Release a new verified package instead.
        if method not in ('GET', 'HEAD', 'OPTIONS') and path.startswith('/admin'):
            if not (path.startswith('/admin/feedback/') or path.startswith('/admin/corrections/')):
                return await respond(403, 'Model training and promotion are offline in the hosted demo.')
        if method not in ('GET', 'HEAD', 'OPTIONS'):
            origin = headers.get('origin')
            expected = os.getenv('PUBLIC_ORIGIN') or os.getenv('RENDER_EXTERNAL_URL') or (scope['scheme'] + '://' + headers.get('host', ''))
            if origin and origin.rstrip('/') != expected.rstrip('/'):
                return await respond(403, 'Cross-site writes are not allowed.')
            if headers.get('sec-fetch-site') == 'cross-site':
                return await respond(403, 'Cross-site writes are not allowed.')
            if headers.get('content-type', '').split(';')[0].lower() != 'application/json':
                return await respond(415, 'Use application/json.')
            # Bound the actual streamed body, including chunked requests.
            body = bytearray()
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    return
                body.extend(message.get('body', b''))
                if len(body) > 65536:
                    return await respond(413, 'Request is too large.')
                if not message.get('more_body', False):
                    break
            delivered = False
            original_receive = receive
            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
                return await original_receive()
            receive = replay
        acquired = False
        try:
            if method == 'POST' and path in ('/analyze', '/chat'):
                try:
                    await asyncio.wait_for(self.slots.acquire(), timeout=0.1)
                    acquired = True
                except asyncio.TimeoutError:
                    return await respond(429, 'Both processing slots are busy. Please try again shortly.', {'Retry-After': '5'})
            await self.app(scope, receive, safe_send)
        finally:
            if acquired:
                self.slots.release()
