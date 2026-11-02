#!/usr/bin/env python3
"""Fail if a model manifest bundles a forbidden licence (research doc: Texo AGPL, IndicOCR hosted terms)."""
import json, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
gates = json.loads((root / "gates.json").read_text())
manifest = root / "models.json"
bad = []
for m in json.loads(manifest.read_text()) if manifest.exists() else []:
    if m.get("bundled") and m["license"] in gates["forbidden_licenses_bundled"]:
        bad.append(m["name"])
print("FAIL bundled forbidden licence:", bad) if bad else print("PASS licences")
sys.exit(1 if bad else 0)
