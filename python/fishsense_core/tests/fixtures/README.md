# Raw decode fixture

`stage2_sample_crop.dng` is a **640×480 crop of a real Olympus sensor mosaic**,
rewrapped as an uncompressed DNG. It exists so the decode chain in
`fishsense_core.image` can be pinned against real underwater sensor data
instead of a mocked `rawpy` returning a seeded random array — see
`tests/test_decode_golden.py`.

## Provenance

Cut from `stage2_sample.ORF` in the data-processing worker's test suite
(`fishsense-lite/services/fishsense-data-processing-workflow-worker/tests/
fixtures/stage2_sample.ORF`), a pool-calibration frame: empty lane, lane rope
across the lower left, open water above.

    crop origin (1200, 1000), 640 × 480, both even so the Bayer phase is kept

Everything the decode reads is carried across verbatim and asserted to
round-trip in `test_fixture_metadata_matches_the_source_sensor`:

| property | value |
|---|---|
| CFA pattern | `[[1, 0], [2, 3]]` over `RGBG` — i.e. GRBG |
| camera white balance | `(2.9375, 1.0, 1.7578, 0.0)` |
| black level / channel | `(266, 266, 266, 267)` |
| white level | `4095` |
| colour matrix | the source's `rgb_xyz_matrix` |

## Why a crop and not the .ORF itself

The .ORF is 15.2 MB and would sit in this repository's history forever. The
crop is 600 KB and keeps everything that matters for a decode test: real shot
noise, real black levels, the real topside-daylight white balance applied
through several metres of water, and the resulting cyan cast with a starved
red channel. On this crop production's `equalize_adapthist` turns the open
water into visible chroma speckle — the frame demonstrates the behaviour the
decode work is about.

What it does *not* pin is Olympus-specific handling inside LibRaw: the fixture
is read through LibRaw's DNG path, not its ORF path. That is the intended
trade. The golden hashes exist to catch changes to *our* chain, not to
LibRaw's.

## Regenerating

`make_fixture.py` rebuilds it from the source .ORF. It needs `tifffile`
(arrives with scikit-image) and a checkout of `fishsense-lite`:

    python tests/fixtures/make_fixture.py \
        --source ~/Repos/school/e4e/fishsense/fishsense-lite/services/\
    fishsense-data-processing-workflow-worker/tests/fixtures/stage2_sample.ORF

Regenerating changes the decoded bytes, so the golden hashes in
`tests/test_decode_golden.py` have to be re-pinned with it. Don't regenerate
casually — the point of the fixture is that it does not move.
