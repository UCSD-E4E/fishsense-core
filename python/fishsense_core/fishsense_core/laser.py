"""Functions for controlling the laser."""
from fishsense_core import _native

calibrate_laser = _native.laser.calibrate_laser # pylint: disable=c-extension-no-member
