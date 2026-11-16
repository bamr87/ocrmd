""".env.example is the template for .env (fleet standard UPS-REPO-32): every variable it names is one ocrmd reads,
nothing in it is live when copied, and a variable the code starts reading by name has to be added to it."""
import re
from pathlib import Path

from ocrmd.config import Settings

ROOT = Path(__file__).resolve().parent.parent
ENTRY = re.compile(r"^(#?)([A-Z][A-Z0-9_]*)=(.*)$")       # `#NAME=value` (commented) or `NAME=value`
READ = re.compile(r"""os\.environ(?:\.get\(|\[)\s*["']([A-Z][A-Z0-9_]*)["']""")
NOT_OURS = {"USER", "HOME"}   # the OS's own: `review --who` defaults to $USER


def _entries() -> dict[str, tuple[bool, str]]:
    out = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if m := ENTRY.match(line):
            out[m.group(2)] = (m.group(1) == "#", m.group(3))
    return out


def _sources() -> list[str]:
    return [p.read_text(encoding="utf-8") for p in sorted((ROOT / "ocrmd").rglob("*.py"))]


def test_nothing_is_live_and_no_credential_has_a_value():
    entries = _entries()
    assert entries, ".env.example names no variables"
    live = [n for n, (commented, _) in entries.items() if not commented]
    assert not live, f"keep every line commented (an empty value copied into .env is still set): {live}"
    valued = [n for n, (_, v) in entries.items() if n.endswith(("_KEY", "_TOKEN")) and v.strip()]
    assert not valued, f"a credential placeholder must stay empty: {valued}"


def test_every_entry_is_something_ocrmd_reads():
    code = "\n".join(_sources())
    for name in _entries():
        if name.startswith("OCRMD_"):
            assert name.removeprefix("OCRMD_").lower() in Settings.model_fields, f"{name} is not a Settings field"
        else:
            assert re.search(rf"\b{name}\b", code), f"{name} is in .env.example but nothing in ocrmd/ reads it"


def test_every_variable_read_by_name_is_listed():
    reads = {n for src in _sources() for n in READ.findall(src)}
    assert reads, "found no environment reads in ocrmd/ — the pattern no longer matches the code"
    missing = sorted(reads - NOT_OURS - set(_entries()))
    assert not missing, f"ocrmd reads {missing} but .env.example does not list them"
