from __future__ import annotations

import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
for path in (
    PACKAGE / "src",
    REPO / "research" / "cliquet_feasibility" / "src",
    REPO / "research" / "cliquet_integration" / "src",
):
    sys.path.insert(0, str(path))
