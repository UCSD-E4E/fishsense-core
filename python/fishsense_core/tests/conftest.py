"""Test-suite-wide setup.

On CI we've seen `pytest` wedge during ONNX Runtime session creation
(`FishSegmentation.load_model()`): ORT's thread-pool work runs behind a C
call that doesn't release the GIL, so when it stalls the interpreter stalls
with it — `pytest-timeout`'s watchdog thread can't even acquire the GIL to
report. `faulthandler`'s watchdog walks the frame stack from C without the
GIL, so it *can* dump every thread and bail. Arm it only on CI.

The dump goes to a real file (``FISHSENSE_HANG_DUMP`` if set) so it survives
``os._exit`` — pytest's fd-level output capture would otherwise swallow it.
The workflow has a step that ``cat``s that file on failure.
"""

import faulthandler
import os
import sys

if os.environ.get("CI"):
    _dump_path = os.environ.get("FISHSENSE_HANG_DUMP")
    _dump_file = open(_dump_path, "w", buffering=1) if _dump_path else sys.stderr  # noqa: SIM115
    # Generous — the whole suite is ~3s; anything past 4 min is a hang.
    faulthandler.dump_traceback_later(240, exit=True, file=_dump_file)
