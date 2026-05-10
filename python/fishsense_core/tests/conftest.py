"""Test-suite-wide setup.

On CI we've seen `pytest` wedge during ONNX Runtime session creation
(`FishSegmentation.load_model()`): ORT's thread-pool work runs behind a C
call that doesn't release the GIL, so when it stalls the interpreter stalls
with it — `pytest-timeout`'s watchdog thread can't even acquire the GIL to
report. `faulthandler`'s watchdog walks the frame stack from C without the
GIL, so it *can* dump every thread and bail. Arm it only on CI so a future
hang prints a usable traceback instead of burning to the job timeout.
"""

import faulthandler
import os

if os.environ.get("CI"):
    # Generous — the whole suite is ~3s; anything past 4 min is a hang.
    faulthandler.dump_traceback_later(240, exit=True)
