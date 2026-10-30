"""Guard historical artifacts from accidental edits during future refactors."""
import subprocess
import sys
from pathlib import Path


def test_frozen_baseline_artifact_checksums():
    repo = Path(__file__).resolve().parent.parent
    check = subprocess.run(
        [sys.executable, str(repo / "scripts" / "verify_baseline.py")],
        cwd=repo, capture_output=True, text=True,
    )
    assert check.returncode == 0, check.stdout + check.stderr
