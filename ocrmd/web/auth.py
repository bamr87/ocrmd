"""Who may talk to the web server.

Three access modes, chosen by how `ocrmd serve` was started:

- `local` (the default): bound to loopback, so only this machine can connect; nothing is checked, as
  before the phone app existed.
- `paired` (`serve --lan`): the port is open to the local network for the phone app. A request from
  this machine still needs nothing — the browser UI keeps working — but every other request must carry
  `Authorization: Bearer <token>` of a paired, unrevoked device (`devices.py`), or it is a 401. The
  only routes open without one are `GET /api/v1/ping` (is this an ocrmd server?) and `POST /api/v1/pair`
  (exchange a pairing code made on this machine for a token).
- `open` (`serve --host … --allow-remote`): the old trusted-network escape hatch; nothing is checked.

In every mode, managing devices (listing, pairing, revoking) is for this machine only (`require_local`):
a paired phone cannot mint or revoke credentials. A token, when sent, is checked in every mode, so a
device is always attributed (a capture's `source`, a sign-off's reviewer) and a revoked token is a 401
rather than a silent success.

"This machine" is a loopback peer, or a peer whose address is the one the connection arrived on (the
browser on this machine dialling its own LAN address). uvicorn rewrites the peer from `X-Forwarded-For`
only for proxies on 127.0.0.1, so a remote client cannot claim to be local; a tunnel or reverse proxy
running *on this machine* without that header would make every request look local — do not put one in
front of `--lan`.
"""
from __future__ import annotations
import json, logging

from fastapi import HTTPException, Request

from .. import devices as devices_mod
from ..devices import is_loopback
from ..store import Store

log = logging.getLogger("ocrmd.web.auth")
MODES = ("local", "paired", "open")
PUBLIC = {("GET", "/api/v1/ping"), ("POST", "/api/v1/pair")}


def bearer(headers: list[tuple[bytes, bytes]]) -> str | None:
    for k, v in headers:
        if k.lower() == b"authorization":
            scheme, _, token = v.decode("latin-1").partition(" ")
            if scheme.lower() == "bearer" and token.strip():
                return token.strip()
    return None


class AccessMiddleware:
    """Pure ASGI (not `BaseHTTPMiddleware`), so the MJPEG preview and file responses stream untouched."""

    def __init__(self, app, st: Store, mode: str = "local"):
        if mode not in MODES:
            raise ValueError(f"access mode must be one of {MODES}")
        self.app, self.st, self.mode = app, st, mode

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        client, server = scope.get("client"), scope.get("server")
        # the same machine dialling its own LAN address arrives from that address, not from 127.0.0.1
        local = is_loopback(client[0] if client else None) or bool(client and server and client[0] == server[0])
        token = bearer(scope.get("headers") or [])
        device = devices_mod.authenticate(self.st, token) if token else None
        state = scope.setdefault("state", {})
        state["local"], state["device"], state["access"] = local, device, self.mode
        if token and device is None:
            return await _deny(send, 401, "this device's token is not valid here (revoked, or paired with another "
                                          "ocrmd); pair it again")
        if self.mode == "paired" and not local and device is None and (scope["method"], scope["path"]) not in PUBLIC:
            return await _deny(send, 401, "pair this device first: run `ocrmd pair` on the computer running ocrmd "
                                          "and enter the code in the app")
        return await self.app(scope, receive, send)


async def _deny(send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                            (b"www-authenticate", b"Bearer")]})
    await send({"type": "http.response.body", "body": body})


def device_of(request: Request) -> devices_mod.Device | None:
    return getattr(request.state, "device", None)


def require_local(request: Request) -> None:
    """A FastAPI dependency: this route is for the machine ocrmd runs on, never a paired device."""
    if not getattr(request.state, "local", False) or device_of(request) is not None:
        raise HTTPException(403, "only the computer running ocrmd can manage paired devices")
