"""Paired devices: how the phone app is let in.

The browser UI trusts whoever reaches it, which is safe because `ocrmd serve` binds to loopback. The
phone app needs the port open to the local network (`ocrmd serve --lan`), and from then on every
request that does not come from this machine must carry a device token (`web/auth.py`). A device gets
its token once, by exchanging a short pairing code the operator made *on this machine* (`ocrmd pair`,
or Settings in the web UI) before the code expires. Nothing else mints a token, and a paired device
cannot pair another.

- Only hashes are stored. The token is handed to the device in the one response that creates it and
  is never printed or logged here; revoking a device keeps its row as the record that it existed.
- Codes are 8 characters from a 31-letter alphabet with no look-alikes (31^8 ~ 8.5e11), live 10
  minutes and work once. `ClaimGuard` voids every open code after 10 wrong guesses, so the space
  cannot be walked from the LAN in the time a code is alive.
- Tokens are 32 random bytes prefixed `ocrmd_`, so a secret scanner can recognise one that leaks.
"""
from __future__ import annotations
import datetime as dt, hashlib, ipaddress, secrets, socket, threading, time, uuid
from dataclasses import dataclass, asdict
from urllib.parse import quote, urlencode

from .store import Store

PAIR_TTL = 600                                  # seconds a pairing code stays claimable
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I/L: read off a screen and typed on a phone
CODE_LENGTH = 8
TOKEN_PREFIX = "ocrmd_"
TOUCH_EVERY = 60                                # seconds between last_seen_at writes for one device
LINK_SCHEME = "ocrmd"                           # the app registers ocrmd:// (mobile/app.json `scheme`)


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def normalize_code(code: str) -> str:
    """What a person typed → the code as generated: case, spaces and dashes do not matter."""
    return "".join(ch for ch in code.upper() if ch.isalnum())


def display_code(code: str) -> str:
    return f"{code[:4]}-{code[4:]}"


@dataclass
class Device:
    id: str
    name: str | None
    platform: str | None
    created_at: str
    paired_at: str | None
    last_seen_at: str | None
    revoked_at: str | None
    pending: bool                               # a pairing code waiting to be claimed

    @classmethod
    def from_row(cls, row) -> "Device":
        return cls(id=row["id"], name=row["name"], platform=row["platform"], created_at=row["created_at"],
                   paired_at=row["paired_at"], last_seen_at=row["last_seen_at"], revoked_at=row["revoked_at"],
                   pending=row["token_hash"] is None and row["revoked_at"] is None
                   and (row["code_expires"] or 0) >= time.time())

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        state = ("revoked" if self.revoked_at else "waiting for the code" if self.pending
                 else "code expired" if self.paired_at is None else f"last seen {self.last_seen_at or 'never'}")
        return f"{self.id}  {(self.name or '—')[:28]:<28} {(self.platform or ''):<8} {state}"


@dataclass
class Pairing:
    id: str
    code: str                                   # shown to the operator once; only its hash is stored
    expires_at: float

    @property
    def display(self) -> str:
        return display_code(self.code)


def start_pairing(st: Store, name: str | None = None, ttl: int = PAIR_TTL) -> Pairing:
    code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
    p = Pairing(id=uuid.uuid4().hex[:10], code=code, expires_at=time.time() + ttl)
    st.add_pairing(p.id, (name or "").strip() or None, hash_secret(code), p.expires_at)
    return p


def claim(st: Store, code: str, name: str, platform: str | None = None) -> tuple[Device, str] | None:
    """Exchange a pairing code for a token. Returns (device, token) — the only time the token exists in
    the clear on this side — or None for a wrong, expired or already-used code."""
    norm = normalize_code(code)
    if len(norm) != CODE_LENGTH:
        return None
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    row = st.claim_pairing(hash_secret(norm), time.time(), hash_secret(token), (name or "").strip()[:64] or "device",
                           (platform or "").strip()[:16] or None)
    return (Device.from_row(row), token) if row is not None else None


def authenticate(st: Store, token: str) -> Device | None:
    if not token or not token.startswith(TOKEN_PREFIX):
        return None
    row = st.device_by_token(hash_secret(token))
    if row is None:
        return None
    seen = row["last_seen_at"]
    try:
        stale = seen is None or (dt.datetime.now().astimezone() - dt.datetime.fromisoformat(seen)).total_seconds() > TOUCH_EVERY
    except ValueError:
        stale = True
    if stale:
        st.touch_device(row["id"])
    return Device.from_row(row)


def list_devices(st: Store, include_revoked: bool = False) -> list[Device]:
    """Paired devices and live codes; with `include_revoked`, also revoked devices and expired codes."""
    out = [Device.from_row(r) for r in st.devices(include_revoked)]
    return out if include_revoked else [d for d in out if d.pending or d.paired_at]


def get(st: Store, device_id: str) -> Device | None:
    """A device by id, revoked or not."""
    return next((d for d in list_devices(st, include_revoked=True) if d.id == device_id), None)


def find(st: Store, ref: str) -> Device:
    """A device by id, or by name when exactly one active device has it."""
    devs = list_devices(st)
    by_id = [d for d in devs if d.id == ref]
    if by_id:
        return by_id[0]
    by_name = [d for d in devs if d.name and d.name.lower() == ref.strip().lower()]
    if len(by_name) == 1:
        return by_name[0]
    if by_name:
        raise LookupError(f"{len(by_name)} devices are named {ref!r}; use the id: "
                          + ", ".join(d.id for d in by_name))
    raise LookupError(f"no device {ref!r}")


def revoke(st: Store, ref: str) -> Device:
    dev = find(st, ref)
    st.revoke_device(dev.id)
    return dev


class ClaimGuard:
    """Counts wrong pairing codes; past `limit` in a row, every open code is voided."""

    def __init__(self, limit: int = 10):
        self.limit, self.misses = limit, 0
        self.lock = threading.Lock()

    def failed(self, st: Store) -> bool:
        """Record a wrong code. True when this one tripped the guard (the open codes are now void)."""
        with self.lock:
            self.misses += 1
            if self.misses < self.limit:
                return False
            self.misses = 0
        st.cancel_pairings()
        return True

    def succeeded(self) -> None:
        with self.lock:
            self.misses = 0


# --- reaching this machine from the phone ------------------------------------------------------------
def is_loopback(host: str | None) -> bool:
    """True for `localhost` and any loopback address (127.0.0.0/8, ::1, ::ffff:127.x). A name is not resolved:
    anything that is not provably loopback counts as remote, so a typo errs toward refusing."""
    if not host:
        return False
    h = host.strip().strip("[]").lower()
    if h == "localhost" or h.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    return ip.is_loopback or bool(getattr(ip, "ipv4_mapped", None) and ip.ipv4_mapped.is_loopback)


def lan_address() -> str | None:
    """This machine's address on the local network, as a phone on the same Wi-Fi would dial it. Asks the
    routing table (a UDP `connect` sends nothing), so it is the interface the default route uses."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        addr = s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()
    return None if addr.startswith("127.") or addr == "0.0.0.0" else addr


def server_url(host: str | None, port: int) -> str:
    """The URL a phone should use: `host` when it names one address (a simulator on this machine may well
    want 127.0.0.1), this machine's LAN address when it is a wildcard bind (0.0.0.0, ::) or not given."""
    if host and host not in ("0.0.0.0", "::", ""):
        h = f"[{host}]" if ":" in host else host
    else:
        h = lan_address() or "127.0.0.1"
    return f"http://{h}:{port}"


def pair_link(server: str, code: str) -> str:
    """What the QR code carries: opens the app on its pairing screen with both fields filled in."""
    return f"{LINK_SCHEME}://pair?" + urlencode({"server": server, "code": code}, quote_via=quote)


def qr_terminal(text: str) -> str | None:
    """The link as a QR code drawn in text, or None when the optional `segno` is not installed."""
    try:
        import io, segno
    except ImportError:
        return None
    buf = io.StringIO()
    segno.make(text, error="m").terminal(out=buf, compact=True)
    return buf.getvalue()


def qr_svg(text: str) -> str | None:
    try:
        import io, segno
    except ImportError:
        return None
    buf = io.BytesIO()
    segno.make(text, error="m").save(buf, kind="svg", scale=5, border=2, xmldecl=False, svgns=True)
    return buf.getvalue().decode("utf-8")
