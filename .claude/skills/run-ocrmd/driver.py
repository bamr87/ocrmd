#!/usr/bin/env python3
"""Drive ocrmd end to end: synthetic pages -> ingest -> OCR -> build -> web API -> screenshots,
plus an optional live-webcam check and a real LLM pass.

Everything runs against a throwaway OCRMD_ROOT (default /tmp/ocrmd-drive) so the repo's own
store (raw/, processed/, ocr/, llm/, documents/, ocrmd.db) is never touched.

    .venv/bin/python .claude/skills/run-ocrmd/driver.py all          # offline: pipeline + api + shots
    .venv/bin/python .claude/skills/run-ocrmd/driver.py pipeline     # ingest + OCR + build, no network
    .venv/bin/python .claude/skills/run-ocrmd/driver.py llm          # real Claude call (costs tokens)
    .venv/bin/python .claude/skills/run-ocrmd/driver.py api          # serve + curl the JSON API
    .venv/bin/python .claude/skills/run-ocrmd/driver.py shots        # serve + playwright screenshots
    .venv/bin/python .claude/skills/run-ocrmd/driver.py camera       # 8 s live webcam scan, OCR-only
    .venv/bin/python .claude/skills/run-ocrmd/driver.py serve        # serve in the foreground, print the URL

Exit code is 0 only if every assertion held.
"""
from __future__ import annotations
import argparse, json, os, shutil, signal, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VENV_BIN = REPO / ".venv" / "bin"
OCRMD = VENV_BIN / "ocrmd"
PY = VENV_BIN / "python"
DOC = "smoke"
PLAYWRIGHT = "playwright@1.56.1"   # any recent version; --channel chrome means no browser download
CHANNEL = "chrome"                 # /Applications/Google Chrome.app — see SKILL.md if you don't have it

# Two pages with *different* body text. Identical text trips the dedup gate (token overlap >= 0.85)
# and the second page is silently dropped as a duplicate.
BODIES = [
    ["It is a truth universally acknowledged that a single man in",
     "possession of a good fortune must be in want of a wife.",
     "However little known the feelings or views of such a man may",
     "be on his first entering a neighbourhood this truth is so well",
     "fixed in the minds of the surrounding families that he is",
     "considered as the rightful property of some one or other of",
     "their daughters."],
    ["My dear Mr Bennet said his lady to him one day have you heard",
     "that Netherfield Park is let at last. Mr Bennet replied that he",
     "had not. But it is returned she for Mrs Long has just been here",
     "and she told me all about it. Mr Bennet made no answer.",
     "Do you not want to know who has taken it cried his wife",
     "impatiently. You want to tell me and I have no objection to",
     "hearing it. This was invitation enough."],
]


def log(msg: str) -> None:
    print(f"[driver] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[driver] FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def run(args, env=None, check=True, **kw) -> subprocess.CompletedProcess:
    p = subprocess.run([str(a) for a in args], capture_output=True, text=True,
                       env={**os.environ, **(env or {})}, **kw)
    if p.stdout:
        print(p.stdout, end="")
    if check and p.returncode != 0:
        print(p.stderr, file=sys.stderr)
        die(f"{args[0]} exited {p.returncode}")
    return p


# --------------------------------------------------------------------------------------- fixtures
def fixtures(dest: Path) -> Path:
    """Render pages that look like a webcam capture: paper on a dark desk, with a margin."""
    import cv2, numpy as np
    from PIL import Image, ImageDraw, ImageFont
    dest.mkdir(parents=True, exist_ok=True)
    try:                                     # a real serif face; PIL's bitmap default OCRs badly
        body_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 46)
        small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 32)
    except OSError:
        body_font = small_font = ImageFont.load_default()
    for i, lines in enumerate(BODIES, 1):
        img = Image.new("RGB", (1500, 2000), (246, 243, 236))
        d = ImageDraw.Draw(img)
        d.text((150, 120), "Chapter 1 — First Impressions", font=small_font, fill="black")
        y = 260
        for line in lines * 3:
            d.text((150, y), line, font=body_font, fill="black")
            y += 78
        d.text((730, 1900), str(11 + i), font=small_font, fill="black")   # printed page numbers 12, 13
        page = np.array(img)[:, :, ::-1].copy()
        h, w = page.shape[:2]
        m = 110
        desk = np.full((h + 2 * m, w + 2 * m, 3), (55, 55, 58), np.uint8)   # dark desk: page detection needs contrast
        desk[m:m + h, m:m + w] = page
        cv2.imwrite(str(dest / f"page-{i:02d}.png"), desk)
    log(f"fixtures: {len(BODIES)} page(s) in {dest}")
    return dest


# --------------------------------------------------------------------------------------- pipeline
def pipeline(root: Path, fresh: bool = True) -> None:
    if fresh and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    env = {"OCRMD_ROOT": str(root)}
    src = fixtures(root / "fixtures")

    log("ocrmd doctor")
    run([OCRMD, "doctor"], env=env)

    log("ocrmd ingest --no-llm  (capture + local OCR only, no network)")
    p = run([OCRMD, "ingest", src, "--doc", DOC, "--no-llm", "--title", "Smoke Test", "--tag", "driver"], env=env)
    if "duplicate:" in p.stdout:
        die("a fixture page was dropped as a duplicate — the two bodies are too similar")

    log("ocrmd status")
    run([OCRMD, "status", "--doc", DOC, "--all"], env=env)

    log("ocrmd build")
    run([OCRMD, "build", "--doc", DOC], env=env)

    pages = sorted((root / "documents" / DOC / "pages").glob("*.md"))
    if len(pages) != len(BODIES):
        die(f"expected {len(BODIES)} page file(s), found {len(pages)}")
    first = pages[0].read_text()
    for needle in ("verification:", "status:", "ocr:", "source:"):
        if needle not in first:
            die(f"page frontmatter missing {needle!r}")
    if "truth universally acknowledged" not in first.lower().replace("\n", " "):
        die("OCR did not read the fixture text — check that an OCR engine is installed (`ocrmd doctor`)")
    built = root / "documents" / DOC / f"{DOC}.md"
    if not built.exists():
        die(f"build produced no {built}")
    log(f"pipeline OK: {len(pages)} page file(s), {built.stat().st_size} B document")
    print(first[:700])


def llm(root: Path) -> None:
    """Run the real LLM pass over the smoke document. Costs tokens."""
    env = {"OCRMD_ROOT": str(root)}
    log("ocrmd verify  (real provider call)")
    run([OCRMD, "verify", "--doc", DOC], env=env)
    run([OCRMD, "status", "--doc", DOC, "--all"], env=env)
    run([OCRMD, "doctor", "--costs"], env=env)


# -------------------------------------------------------------------------------------------- web
class Server:
    def __init__(self, root: Path, port: int):
        self.root, self.port, self.proc = root, port, None
        self.log_path = root / "serve.log"
        self.url = f"http://127.0.0.1:{port}"

    def __enter__(self):
        self.fh = self.log_path.open("w")
        self.proc = subprocess.Popen([str(OCRMD), "serve", "--port", str(self.port), "--no-open"],
                                     stdout=self.fh, stderr=subprocess.STDOUT,
                                     env={**os.environ, "OCRMD_ROOT": str(self.root)},
                                     start_new_session=True)
        for _ in range(120):                    # uvicorn is up in ~1.5 s; first import of cv2/pyobjc dominates
            if self.proc.poll() is not None:
                die(f"serve exited early:\n{self.log_path.read_text()}")
            try:
                urllib.request.urlopen(f"{self.url}/api/documents", timeout=1).read()
                log(f"serve ready at {self.url} (pid {self.proc.pid})")
                return self
            except Exception:
                time.sleep(0.25)
        die(f"serve never answered on {self.url}:\n{self.log_path.read_text()}")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            # uvicorn.run() installs its own SIGINT handler; SIGTERM to the group is what actually stops it
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        self.fh.close()
        log("serve stopped")

    def get(self, path: str):
        with urllib.request.urlopen(f"{self.url}{path}", timeout=30) as r:
            body = r.read()
        return json.loads(body) if r.headers.get_content_type() == "application/json" else body

    def post(self, path: str, payload: dict | None = None):
        data = json.dumps(payload or {}).encode()
        req = urllib.request.Request(f"{self.url}{path}", data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())


def api(root: Path, port: int) -> None:
    if not (root / "documents" / DOC).exists():
        die(f"no {DOC} document in {root} — run `pipeline` first")
    with Server(root, port) as s:
        docs = s.get("/api/documents")
        log(f"GET /api/documents -> {json.dumps(docs)[:200]}")
        if not any(d["slug"] == DOC for d in docs):
            die(f"{DOC} missing from /api/documents")

        detail = s.get(f"/api/documents/{DOC}")
        rows = detail["page_list"]          # "pages" is an int count; the table is under "page_list"
        n = len(rows)
        log(f"GET /api/documents/{DOC} -> {n} page(s), statuses={[r['status'] for r in rows]}")
        if n != len(BODIES):
            die(f"expected {len(BODIES)} pages, API reports {n}")

        page = s.get(f"/api/documents/{DOC}/pages/1")
        log(f"GET .../pages/1 -> {len(page['body'])} chars of markdown, status={page['status']}")

        png = s.get(f"/api/documents/{DOC}/pages/1/image/processed")
        if png[:4] != b"\x89PNG":
            die("page image endpoint did not return a PNG")
        log(f"GET .../pages/1/image/processed -> {len(png)} B PNG")

        flat = s.get(f"/api/documents/{DOC}/pages/1/image/flat")
        if flat[:4] != b"\x89PNG":
            die("the flat view did not return a PNG")
        log(f"GET .../pages/1/image/flat -> {len(flat)} B PNG")

        # the region tool: drag a box on the page image, read that box on its own
        region = s.post(f"/api/documents/{DOC}/pages/1/region", {"x": 0.05, "y": 0.05, "w": 0.9, "h": 0.35})
        if not region["text"].strip():
            die("region read came back empty")
        log(f"POST .../pages/1/region -> {region['size'][0]}x{region['size'][1]}px enlarged "
            f"{region['scale']}x, {len(region['text'].split())} word(s)")

        # A UI edit must be preserved by a later verify: the write deliberately does NOT refresh the
        # pipeline fingerprint, which is what marks the file as hand-edited.
        edited = page["body"].rstrip() + "\n\nEDITED-BY-DRIVER\n"
        req = urllib.request.Request(f"{s.url}/api/documents/{DOC}/pages/1",
                                     data=json.dumps({"body": edited}).encode(), method="PUT",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
        s.post(f"/api/documents/{DOC}/pages/1/signoff", {"who": "driver"})   # field is `who`, not `reviewer`
        after = s.get(f"/api/documents/{DOC}/pages/1")
        if after["status"] != "reviewed" or "EDITED-BY-DRIVER" not in after["body"]:
            die(f"edit + signoff did not stick: status={after['status']}")
        log("PUT body + POST signoff -> status=reviewed, edit present")

        job = s.post("/api/jobs", {"kind": "build", "doc": DOC})   # the field is `doc`; kinds: verify|build|reprocess|ingest
        for _ in range(120):
            st = s.get(f"/api/jobs/{job['id']}")
            if st["status"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.5)
        log(f"POST /api/jobs build -> {st['status']}")
        if st["status"] != "done":
            die(f"build job ended {st['status']}: {st.get('log')}")

        body = (root / "documents" / DOC / "pages" / "0001.md").read_text()
        if "EDITED-BY-DRIVER" not in body:
            die("the build job overwrote a human edit")

        rep = s.get("/api/doctor")
        log(f"GET /api/doctor -> ocr={rep['ocr_engine']} db={rep['db_integrity']} cameras={len(rep['cameras'])}")
    log("api OK")


def shots(root: Path, port: int, out: Path) -> None:
    if not shutil.which("npx"):
        die("npx not found — needed for `npx playwright screenshot`")
    out.mkdir(parents=True, exist_ok=True)
    # (name, hash fragment, selector that only exists once the view has rendered).
    # Page route is #/doc/<slug>/page/<seq> — #/doc/<slug>/<seq> silently falls through to the
    # document view, so a screenshot taken there looks plausible and is the wrong screen.
    views = [
        ("dashboard", "", ".cards .card"),
        ("document",  f"#/doc/{DOC}", "#bulkbar"),
        ("page",      f"#/doc/{DOC}/page/1", "#body"),
        ("scan",      "#/scan", "#gate"),
        ("jobs",      "#/jobs", "h1"),
        ("settings",  "#/settings", "#m-doc"),
    ]
    with Server(root, port) as s:
        for name, frag, sel in views:
            path = out / f"{name}.png"
            path.unlink(missing_ok=True)
            # --channel chrome drives the installed Google Chrome: no ~150 MB browser download and no
            # playwright-version-vs-cached-revision mismatch. --wait-for-selector is the real assertion —
            # `load` fires on the empty shell, so without it a broken route still yields a pretty PNG.
            p = run(["npx", "--yes", PLAYWRIGHT, "screenshot", "--channel", CHANNEL, "--full-page",
                     f"--wait-for-selector={sel}", "--wait-for-timeout=1200",
                     f"{s.url}/{frag}", str(path)], check=False)
            if p.returncode != 0 or not path.exists():
                print(p.stderr, file=sys.stderr)
                die(f"screenshot {name} failed (selector {sel!r} never appeared)")
            log(f"shot {name}: {path} ({path.stat().st_size} B)")
    log(f"shots OK -> {out}")


def camera(root: Path, port: int, out: Path, seconds: float) -> None:
    """Prove the capture lane works: start a real scan through the API, watch the gates, grab a frame.

    OCR-only and time-boxed. `duration` is enforced inside the session, so even if this script dies
    the scan stops on its own instead of leaving the camera open.
    """
    out.mkdir(parents=True, exist_ok=True)
    with Server(root, port) as s:
        cams = s.get("/api/cameras")
        log(f"GET /api/cameras -> {[c['name'] for c in cams]}")
        if not cams:
            die("no camera reported by /api/cameras — `ocrmd doctor` says whether macOS denied access")
        snap = s.post("/api/scan/start", {"doc": "camera-check", "camera": cams[0]["index"],
                                          "focus": "continuous", "no_llm": True,
                                          "duration": seconds, "title": "Camera check"})
        if not snap.get("running"):
            die(f"scan did not start: {json.dumps(snap)[:400]}")
        log(f"POST /api/scan/start -> running on {snap.get('camera', {}).get('name')}")
        gates = []
        deadline = time.time() + seconds
        while time.time() < deadline:
            g = s.get("/api/scan")
            if not g.get("running"):
                break
            gate = (g.get("gate") or {})
            gates.append(gate.get("gate"))
            time.sleep(0.5)
        frame = out / "scan-frame.jpg"
        try:
            frame.write_bytes(s.get("/api/scan/frame.jpg"))
            log(f"GET /api/scan/frame.jpg -> {frame} ({frame.stat().st_size} B)")
        except urllib.error.HTTPError as e:
            die(f"no live frame: {e}")
        s.post("/api/scan/stop")
        for _ in range(40):
            if not s.get("/api/scan").get("running"):
                break
            time.sleep(0.5)
        else:
            die("scan did not stop")
        seen = [x for x in dict.fromkeys(gates) if x]
        log(f"gates seen: {seen or ['(none — camera produced no gated frames)']}")
        if not seen:
            die("the capture loop never reported a gate state")
        # The gates may well have fired: whatever was under the camera is now pages on disk. Bin it.
        req = urllib.request.Request(f"{s.url}/api/documents/camera-check?confirm=camera-check",
                                     method="DELETE")
        try:
            log(f"cleanup: {json.loads(urllib.request.urlopen(req, timeout=60).read())['summary']}")
        except urllib.error.HTTPError as e:
            log(f"cleanup skipped ({e.code}) — captures remain under {root}")
    log("camera OK")


def serve(root: Path, port: int) -> None:
    log(f"serving {root} on http://127.0.0.1:{port} (Ctrl-C to stop)")
    os.execve(str(OCRMD), [str(OCRMD), "serve", "--port", str(port), "--no-open"],
              {**os.environ, "OCRMD_ROOT": str(root)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fixtures", "pipeline", "llm", "api", "shots", "camera", "serve", "all"])
    ap.add_argument("--root", default="/tmp/ocrmd-drive", help="throwaway OCRMD_ROOT (default: %(default)s)")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--shots-dir", default="/tmp/ocrmd-shots")
    ap.add_argument("--seconds", type=float, default=8.0, help="camera: how long to hold the scan open")
    ap.add_argument("--keep", action="store_true", help="reuse an existing --root instead of wiping it")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.shots_dir)
    if not OCRMD.exists():
        die(f"{OCRMD} not found — run: python3 -m venv .venv && .venv/bin/pip install -e '.[dev,rapidocr]'")
    if a.command == "fixtures":
        fixtures(root / "fixtures")
    elif a.command == "pipeline":
        pipeline(root, fresh=not a.keep)
    elif a.command == "llm":
        llm(root)
    elif a.command == "api":
        api(root, a.port)
    elif a.command == "shots":
        shots(root, a.port, out)
    elif a.command == "camera":
        camera(root, a.port, out, a.seconds)
    elif a.command == "serve":
        serve(root, a.port)
    elif a.command == "all":
        pipeline(root, fresh=not a.keep)
        api(root, a.port)
        shots(root, a.port, out)
    log("done")


if __name__ == "__main__":
    main()
