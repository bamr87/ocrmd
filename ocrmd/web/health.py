"""Doctor report as data, shared by `ocrmd doctor` and the web UI."""
from __future__ import annotations
from pathlib import Path
from ..config import Settings
from ..store import Store
from .. import ocr, pipeline, llm as llm_mod, sinks as sinks_mod


def cameras(root: Path | None = None) -> list[dict]:
    from .. import camctl, focus
    out = []
    try:
        mapping = camctl.map_cv2_indices()
    except Exception:
        return out
    for i, d in sorted(mapping.items()):
        cam = {"index": i, "name": d, "format": None, "focus_modes": [], "focus": None, "exposure": None,
               "lens_position": False, "warning": None}
        caps = camctl.capabilities(name=d)
        if caps:
            cam["format"] = list(camctl.best_native_format(name=d) or ())
            state = camctl.focus_state(name=d)
            cam["focus_modes"] = caps.focus_modes
            # without this the focus bracket cannot be shot at all — the shutter falls back to the burst
            cam["lens_position"] = bool(caps.custom_lens_position)
            cam["focus"], cam["exposure"] = state.get("focus"), state.get("exposure")
            if state.get("focus") == "locked" and "continuous" in caps.focus_modes:
                cam["warning"] = "locked focus blurs every frame; `scan` sets continuous"
            # what this camera has managed before, remembered across sessions: the anchor the shutter's
            # sharpness floor uses so a session that starts out of focus cannot lower its own bar
            cam["best_sharpness"] = (focus.load_best(root, d) or None) if root else None
        out.append(cam)
    return out


def doctor_report(s: Settings, st: Store, probe_cameras: bool = True) -> dict:
    from ..capture import camera_authorized
    pipeline.backfill_page_meta(s, st)
    missing = pipeline.missing_artifacts(s, st)
    return {
        "root": str(s.root),
        "config": {"model": s.model, "provider": s.provider, "prompt_version": s.prompt_version, "focus": s.focus,
                   "focus_bracket": s.focus_bracket, "llm_workers": s.llm_workers,
                   "log": str(s.log_path) if s.log_path else None},
        "ocr_engine": pipeline.ocr_version(s),          # the configured engine set, e.g. ocr@3:apple_vision+rapidocr
        "ocr_engines_installed": ocr.available_engines(),
        "credentials": {"anthropic": llm_mod.credentials_available("anthropic"), "xai": llm_mod.credentials_available("xai")},
        "dictionary": Path("/usr/share/dict/words").exists(),
        "cameras": cameras(s.root) if probe_cameras else [],
        "camera_authorized": camera_authorized(),
        "db_path": str(s.db_path), "db_integrity": st.integrity(),
        "documents": [{"slug": slug, "pages": sum(st.counts(slug).values()), "counts": st.counts(slug)} for slug in st.documents()],
        "missing_artifacts": missing,
        "orphan_artifacts": len(pipeline.orphan_artifacts(s, st)),
        "open_batches": len(st.open_batches()),
        # destinations are listed, not probed: a receiver that is down must not slow the doctor down
        "sinks": [{"name": x.name, "url": getattr(x, "url", "")} for x in sinks_mod.registry(s)],
        "library": {"documents": len(st.documents()), "pages": len(st.pages()), "indexed": st.indexed_count(),
                    "fts": st.fts, "export_dir": str(s.export_dir)},
        "costs": [{"provider": r["provider"], "model": r["model"], "n": r["n"], "tokens_in": r["tin"] or 0,
                   "tokens_out": r["tout"] or 0, "cache_read": r["cached"] or 0, "avg_latency_ms": int(r["lat"] or 0)}
                  for r in st.costs()],
    }
