"""docs/reference/ is generated from the CLI and the Settings model (tools/gen_reference.py); a command, option
or setting added without regenerating it fails here. The generator runs in its own process because it imports
`ocrmd.cli` against pure defaults, which must not disturb the settings the rest of the suite patches."""
import subprocess, sys
from pathlib import Path
import pytest

GEN = Path(__file__).resolve().parent.parent / "tools" / "gen_reference.py"


def test_the_generated_reference_matches_the_code():
    r = subprocess.run([sys.executable, str(GEN), "--check"], capture_output=True, text=True, timeout=300)
    if r.returncode == 2:   # the generator could not import ocrmd in this interpreter
        pytest.skip(r.stderr.strip())
    assert r.returncode == 0, r.stderr + "\nregenerate with: python3 tools/gen_reference.py --write"
