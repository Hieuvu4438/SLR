from __future__ import annotations

import subprocess
import sys


def test_import_does_not_import_torch_or_open_network_modules() -> None:
    code = (
        "import sys; import ocem; "
        "forbidden={'torch','requests','urllib.request'}; "
        "loaded=forbidden.intersection(sys.modules); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

