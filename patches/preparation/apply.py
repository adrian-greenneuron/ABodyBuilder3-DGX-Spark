"""Apply only the exact preparation API changes qualified on the component branch.

Fail on an unexpected upstream source rather than patching a different version.
The manifest also makes this step idempotent for incremental local builds.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def install():
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, entry in manifest.items():
        target = Path(importlib.util.find_spec(entry["module"]).origin)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest == entry["patched_sha256"]:
            print(f"Already verified: {entry['module']}")
            continue
        if digest != entry["original_sha256"]:
            raise RuntimeError(f"Unexpected {entry['module']} source ({digest}); refusing to patch.")
        subprocess.run(["patch", "--batch", "--forward", str(target), "-i",
                        str(directory / (name + ".patch"))], check=True)
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry["patched_sha256"]:
            raise RuntimeError(f"Patched {entry['module']} does not match the qualified source.")
        print(f"Installed and verified: {entry['module']}")


if __name__ == "__main__":
    install()
