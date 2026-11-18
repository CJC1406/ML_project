"""Versioned model bundle: every trained candidate model, the ensemble meta-learner and the HMM settings.

Layout (models/pathsnap_v1/ by default):
    manifest.json                  versions, features, candidate settings, configs, metrics, checksums
    base/<name>.joblib             each fitted base classifier
    stacking_meta.joblib           logistic-regression meta-learner for the stacking ensemble
    hmm_config.json                tuned HMM/Viterbi parameters
Selectable models after loading: every base model, "soft_vote" and "stacking".
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib

from pathsnap.models.ensemble import CandidateEnsemble

BUNDLE_VERSION = 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_bundle(out: Path, base_models: dict, meta, hmm_config: dict, manifest_extra: dict) -> dict:
    out = Path(out)
    (out / "base").mkdir(parents=True, exist_ok=True)
    files = {}
    for name, model in base_models.items():
        path = out / "base" / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        files[f"base/{name}.joblib"] = sha256(path)
    joblib.dump(meta, out / "stacking_meta.joblib", compress=3)
    files["stacking_meta.joblib"] = sha256(out / "stacking_meta.joblib")
    (out / "hmm_config.json").write_text(json.dumps(hmm_config, indent=2) + "\n")
    files["hmm_config.json"] = sha256(out / "hmm_config.json")
    manifest = dict(bundle_version=BUNDLE_VERSION, base_models=list(base_models),
                    ensembles=["soft_vote", "stacking"], files=files, **manifest_extra)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    return manifest


class BundleError(RuntimeError):
    pass


def load_bundle(path: Path, verify: bool = True):
    """Return (models dict name -> classifier, hmm_config, manifest). Fails clearly when files are absent."""
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise BundleError(f"model bundle not found: {manifest_path} (run scripts/export_model.py)")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("bundle_version") != BUNDLE_VERSION:
        raise BundleError(f"unsupported bundle version {manifest.get('bundle_version')}")
    for rel, digest in manifest["files"].items():
        f = path / rel
        if not f.is_file():
            raise BundleError(f"bundle file missing: {f}")
        if verify and sha256(f) != digest:
            raise BundleError(f"checksum mismatch for {f}; re-export the bundle")
    base = {n: joblib.load(path / "base" / f"{n}.joblib") for n in manifest["base_models"]}
    meta = joblib.load(path / "stacking_meta.joblib")
    models = dict(stacking=CandidateEnsemble(base, "stacking", meta),
                  soft_vote=CandidateEnsemble(base, "soft_vote"), **base)
    hmm_config = json.loads((path / "hmm_config.json").read_text())
    return models, hmm_config, manifest
