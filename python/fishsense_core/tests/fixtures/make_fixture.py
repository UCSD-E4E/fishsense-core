"""Rebuild ``stage2_sample_crop.dng`` from the source .ORF.

Not run by the test suite — the fixture is checked in. This is here so the
fixture is reproducible rather than a mystery blob, and so a future crop (a
reef frame, a frame with a laser dot) can be cut the same way.

See ``README.md`` in this directory for provenance and for what re-running
this obliges you to re-pin.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np

#: Crop origin and size. Both origin coordinates must be even: the Bayer
#: pattern repeats every 2 px, so an odd offset would hand LibRaw a mosaic
#: whose phase disagrees with the CFAPattern tag and silently swap red for
#: green over the whole frame.
CROP = (1200, 1000, 640, 480)

#: DNG's CFAPattern codes, against LibRaw's ``color_desc`` letters. The second
#: green ("G2", index 3 in this sensor's ``raw_pattern``) is still green.
CFA_CODES = {"R": 0, "G": 1, "B": 2}


def _rational(value: float, denominator: int = 1_000_000) -> tuple[int, int]:
    return int(round(value * denominator)), denominator


def build(source: Path, x0: int, y0: int, width: int, height: int) -> bytes:
    """Read `source`, cut the mosaic, and return DNG bytes."""
    import rawpy  # pylint: disable=import-outside-toplevel
    import tifffile  # pylint: disable=import-outside-toplevel

    if x0 % 2 or y0 % 2 or width % 2 or height % 2:
        raise ValueError("crop origin and size must be even to keep the Bayer phase")

    with rawpy.imread(str(source)) as raw:
        mosaic = raw.raw_image_visible[y0 : y0 + height, x0 : x0 + width].copy()
        pattern = np.asarray(raw.raw_pattern)
        color_desc = raw.color_desc.decode()
        black = [int(b) for b in raw.black_level_per_channel]
        white = int(raw.white_level)
        camera_wb = [float(v) for v in raw.camera_whitebalance]
        cam_xyz = np.asarray(raw.rgb_xyz_matrix)[:3]

    cfa = bytes(
        CFA_CODES[color_desc[int(pattern[i, j])]] for i in range(2) for j in range(2)
    )
    # DNG stores the reciprocal of LibRaw's raw multipliers, green pinned to 1.
    neutral = [1.0 / camera_wb[0], 1.0, 1.0 / camera_wb[2]]

    extratags = [
        (33421, 3, 2, (2, 2), False),                      # CFARepeatPatternDim
        (33422, 1, 4, cfa, False),                         # CFAPattern
        (50706, 1, 4, bytes([1, 4, 0, 0]), False),         # DNGVersion
        (50707, 1, 4, bytes([1, 1, 0, 0]), False),         # DNGBackwardVersion
        (50708, 2, 0, "FishSense stage2_sample crop", False),  # UniqueCameraModel
        (50721, 10, 9,                                     # ColorMatrix1 (XYZ->cam)
         tuple(v for x in cam_xyz.ravel() for v in _rational(float(x))), False),
        (50713, 3, 2, (2, 2), False),                      # BlackLevelRepeatDim
        (50714, 3, 4, tuple(black), False),                # BlackLevel
        (50717, 3, 1, (white,), False),                    # WhiteLevel
        (50728, 5, 3,                                      # AsShotNeutral
         tuple(v for x in neutral for v in _rational(x)), False),
        (50778, 3, 1, (21,), False),                       # CalibrationIlluminant1 = D65
    ]

    buffer = io.BytesIO()
    # Uncompressed: this LibRaw build rejects a deflate-compressed DNG
    # outright ("Unsupported file format or not RAW file"), so the 600 KB is
    # not negotiable without a lossless-JPEG writer.
    tifffile.imwrite(
        buffer, mosaic, photometric=32803, planarconfig=None, extratags=extratags
    )
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="the .ORF to crop")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "stage2_sample_crop.dng"
    )
    parser.add_argument("--crop", type=int, nargs=4, default=CROP,
                        metavar=("X0", "Y0", "W", "H"))
    args = parser.parse_args()

    data = build(args.source, *args.crop)
    args.out.write_bytes(data)
    print(f"wrote {args.out} ({len(data)} bytes)")
    print("re-pin the golden hashes in tests/test_decode_golden.py")


if __name__ == "__main__":
    main()
