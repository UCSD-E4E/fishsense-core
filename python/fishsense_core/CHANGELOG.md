# Changelog

## [1.7.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense_core-v1.6.0...fishsense_core-v1.7.0) (2026-04-27)


### Features

* expose snap to depth for python testing ([b901d74](https://github.com/UCSD-E4E/fishsense-core/commit/b901d74cc0e643fa7e8186db1b50412bb2f47199))

## [1.6.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense_core-v1.5.0...fishsense_core-v1.6.0) (2026-04-27)


### Features

* bind WorldPointHandler to Python with project + laser triangulation ([ce5a863](https://github.com/UCSD-E4E/fishsense-core/commit/ce5a863039baf8c0dbc92f1d67cddc39d5ea76c2))


### Bug Fixes

* accept float32 in calibrate_laser PyO3 binding ([fc4b394](https://github.com/UCSD-E4E/fishsense-core/commit/fc4b39456f2823dc1f17b28d781a66889efa4291))
* coerce WorldPointHandler inputs to float64 in the Python wrapper ([b813662](https://github.com/UCSD-E4E/fishsense-core/commit/b81366265f0255371eff9ba3341d80ba37aca479))

## [1.5.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense_core-v1.4.1...fishsense_core-v1.5.0) (2026-04-26)


### Features

* return mask-space coords from find_head_tail_depth ([08577b3](https://github.com/UCSD-E4E/fishsense-core/commit/08577b3640b9cb640ce91756655864248af1a3ca))

## [1.4.1](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense_core-v1.4.0...fishsense_core-v1.4.1) (2026-04-26)


### Miscellaneous Chores

* **fishsense_core:** Synchronize fishsense versions

## [1.4.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense_core-v1.3.0...fishsense_core-v1.4.0) (2026-04-21)


### Miscellaneous Chores

* **fishsense_core:** Synchronize fishsense versions

## [1.3.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense_core-v1.2.0...fishsense_core-v1.3.0) (2026-04-21)


### Features

* add inference_single for single-fish mask output ([ea196dd](https://github.com/UCSD-E4E/fishsense-core/commit/ea196ddeedb9c9f8a7f000a1efd35bc0a20fed79))

## [1.2.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense_core-v1.1.0...fishsense_core-v1.2.0) (2026-04-20)


### Features

* fish head tail detector ([c68ee34](https://github.com/UCSD-E4E/fishsense-core/commit/c68ee34b1f859bb2eb81ffcaaa75f8b9ead98b55))


### Bug Fixes

* don't init the logger if another one is already inited ([012f672](https://github.com/UCSD-E4E/fishsense-core/commit/012f6723e26359668721b968905445eb1b7bc6eb))
* pyo3 submodule issues ([4ef48da](https://github.com/UCSD-E4E/fishsense-core/commit/4ef48da141ada974d0dc459610bdacdd9fdedf73))

## [1.1.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense_core-v1.0.0...fishsense_core-v1.1.0) (2026-04-20)


### Features

* export fish segmentation ([c012e11](https://github.com/UCSD-E4E/fishsense-core/commit/c012e11cd03b49be4ff79c9dd88cbcd4b68c67be))


### Bug Fixes

* pylint errors ([a60bd33](https://github.com/UCSD-E4E/fishsense-core/commit/a60bd33d4841c49ea5949a12b5cfbb341b37bb48))

## [1.0.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense_core-v0.8.0...fishsense_core-v1.0.0) (2026-04-14)


### Miscellaneous Chores

* **fishsense_core:** Synchronize fishsense versions

## [0.8.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense_core-v0.7.0...fishsense_core-v0.8.0) (2026-04-14)


### Features

* add base image file from previous implementation ([57d5a81](https://github.com/UCSD-E4E/fishsense-core/commit/57d5a816e38b028b8b114d8a0b0aefb999604ab4))
* add missing raw image and rectified image ([7ca7800](https://github.com/UCSD-E4E/fishsense-core/commit/7ca78007e9a08dcdc92703af88a06a7116ab2f97))
* additional logging.  gpu-based connected compoennts implementation ([cc18072](https://github.com/UCSD-E4E/fishsense-core/commit/cc18072cb54a12aebb6656f6b232f72c87156f26))
* bring all versions up to date ([2ed4c25](https://github.com/UCSD-E4E/fishsense-core/commit/2ed4c25727f0f7c6982a474b949056056357c4a8))
* expose calibrate laser method ([1a7d903](https://github.com/UCSD-E4E/fishsense-core/commit/1a7d903463944b1af2ec76dc05e775340fb74336))
* initial checkin ([916aa0b](https://github.com/UCSD-E4E/fishsense-core/commit/916aa0b92c99dd9331b124ff753879cf852a2154))
* let's try to properly export laser ([e6aebf8](https://github.com/UCSD-E4E/fishsense-core/commit/e6aebf84080c13e7d50ea000aec2ad5b1e4359bf))
* release please ([2b752d4](https://github.com/UCSD-E4E/fishsense-core/commit/2b752d4e7f7822230f143c1452afc32bb2666625))
* release structure and readme ([dfe694f](https://github.com/UCSD-E4E/fishsense-core/commit/dfe694feb22f4aca44cdf45c4d8298166e4e6c44))
* switch fishsense_core python package to maturin ([8d8b7cb](https://github.com/UCSD-E4E/fishsense-core/commit/8d8b7cb7997dbbd109b1ed24b11117ccad0062ff))


### Bug Fixes

* add missing module file ([1eb64d7](https://github.com/UCSD-E4E/fishsense-core/commit/1eb64d748a7976b36ab10d31d08fad4fb2db58cf))
* create uv.lock in the python directory ([048c221](https://github.com/UCSD-E4E/fishsense-core/commit/048c221afd3c7abde023812e937c5ef9c788076a))
* do calibration in f64 rather than f32 ([24fd90d](https://github.com/UCSD-E4E/fishsense-core/commit/24fd90dfa25af25e88da0f989c1cf5c233b69e9f))
* ensure tha the laser_origin z is on the z=0 plane ([0bd7d72](https://github.com/UCSD-E4E/fishsense-core/commit/0bd7d72c531b2ae16bfcc02d23d85302eb57c0e2))
* export the native module ([2ae0514](https://github.com/UCSD-E4E/fishsense-core/commit/2ae05147b42f947c5505eb15b278677c9a6a5e6a))
* let's try a different module syntax ([d323564](https://github.com/UCSD-E4E/fishsense-core/commit/d3235644d7322cea7b93debc919d6fd7e53fd14a))
* let's try laser module again ([9932a20](https://github.com/UCSD-E4E/fishsense-core/commit/9932a2064ecf8fbfd7864c42044e582decc1cc23))
* let's try updaing the module paths again ([7856fc8](https://github.com/UCSD-E4E/fishsense-core/commit/7856fc872dd437dcb8befed916ba7c48ab2f6aad))
* pylint errors ([71f5c87](https://github.com/UCSD-E4E/fishsense-core/commit/71f5c8743a96caad9b761c005dd698b59fa260bd))
* python dependencies ([6bf2e1c](https://github.com/UCSD-E4E/fishsense-core/commit/6bf2e1c2aceaaa8bc2a7fbacafe0b2c8fab130e7))
* python tests now pass ([47cd7d3](https://github.com/UCSD-E4E/fishsense-core/commit/47cd7d3ec20c16aaaa6fdcdad2240e76cc113fed))
* readd missing files ([f797879](https://github.com/UCSD-E4E/fishsense-core/commit/f797879a31115642eb1840cd1f7ae47da4e7a23f))
* release please releases should be linked ([2760e21](https://github.com/UCSD-E4E/fishsense-core/commit/2760e21747d4d27f23d0bd002138bbacdf2a411c))
* specify native module ([c7589c8](https://github.com/UCSD-E4E/fishsense-core/commit/c7589c8520ade8ec9ee82cfb658b4e4decfdb495))
* split the module ([e8eeb1d](https://github.com/UCSD-E4E/fishsense-core/commit/e8eeb1d2aa9c2269d0637cf35fbfbbcb01657741))
* try changing the module name ([e0a4f1e](https://github.com/UCSD-E4E/fishsense-core/commit/e0a4f1e1a94e5d1aaffe65aac048fdc6b2731705))
* try to export calibrate laser again ([19844ee](https://github.com/UCSD-E4E/fishsense-core/commit/19844eeb4a6412b842007331bfc1e909904d526e))
* try to use a different name for module ([e4c33fd](https://github.com/UCSD-E4E/fishsense-core/commit/e4c33fd52e6127183967960f37e0ca6f22bc8c14))

## [0.5.1](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense_core-v0.5.0...fishsense_core-v0.5.1) (2026-04-14)


### Bug Fixes

* create uv.lock in the python directory ([048c221](https://github.com/UCSD-E4E/fishsense-core/commit/048c221afd3c7abde023812e937c5ef9c788076a))

## [0.5.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense_core-v0.4.0...fishsense_core-v0.5.0) (2026-04-13)


### Features

* add base image file from previous implementation ([57d5a81](https://github.com/UCSD-E4E/fishsense-core/commit/57d5a816e38b028b8b114d8a0b0aefb999604ab4))
* add missing raw image and rectified image ([7ca7800](https://github.com/UCSD-E4E/fishsense-core/commit/7ca78007e9a08dcdc92703af88a06a7116ab2f97))
* additional logging.  gpu-based connected compoennts implementation ([cc18072](https://github.com/UCSD-E4E/fishsense-core/commit/cc18072cb54a12aebb6656f6b232f72c87156f26))
* expose calibrate laser method ([1a7d903](https://github.com/UCSD-E4E/fishsense-core/commit/1a7d903463944b1af2ec76dc05e775340fb74336))
* initial checkin ([916aa0b](https://github.com/UCSD-E4E/fishsense-core/commit/916aa0b92c99dd9331b124ff753879cf852a2154))
* let's try to properly export laser ([e6aebf8](https://github.com/UCSD-E4E/fishsense-core/commit/e6aebf84080c13e7d50ea000aec2ad5b1e4359bf))
* release please ([2b752d4](https://github.com/UCSD-E4E/fishsense-core/commit/2b752d4e7f7822230f143c1452afc32bb2666625))
* release structure and readme ([dfe694f](https://github.com/UCSD-E4E/fishsense-core/commit/dfe694feb22f4aca44cdf45c4d8298166e4e6c44))
* switch fishsense_core python package to maturin ([8d8b7cb](https://github.com/UCSD-E4E/fishsense-core/commit/8d8b7cb7997dbbd109b1ed24b11117ccad0062ff))


### Bug Fixes

* add missing module file ([1eb64d7](https://github.com/UCSD-E4E/fishsense-core/commit/1eb64d748a7976b36ab10d31d08fad4fb2db58cf))
* do calibration in f64 rather than f32 ([24fd90d](https://github.com/UCSD-E4E/fishsense-core/commit/24fd90dfa25af25e88da0f989c1cf5c233b69e9f))
* ensure tha the laser_origin z is on the z=0 plane ([0bd7d72](https://github.com/UCSD-E4E/fishsense-core/commit/0bd7d72c531b2ae16bfcc02d23d85302eb57c0e2))
* export the native module ([2ae0514](https://github.com/UCSD-E4E/fishsense-core/commit/2ae05147b42f947c5505eb15b278677c9a6a5e6a))
* let's try a different module syntax ([d323564](https://github.com/UCSD-E4E/fishsense-core/commit/d3235644d7322cea7b93debc919d6fd7e53fd14a))
* let's try laser module again ([9932a20](https://github.com/UCSD-E4E/fishsense-core/commit/9932a2064ecf8fbfd7864c42044e582decc1cc23))
* let's try updaing the module paths again ([7856fc8](https://github.com/UCSD-E4E/fishsense-core/commit/7856fc872dd437dcb8befed916ba7c48ab2f6aad))
* pylint errors ([71f5c87](https://github.com/UCSD-E4E/fishsense-core/commit/71f5c8743a96caad9b761c005dd698b59fa260bd))
* python dependencies ([6bf2e1c](https://github.com/UCSD-E4E/fishsense-core/commit/6bf2e1c2aceaaa8bc2a7fbacafe0b2c8fab130e7))
* python tests now pass ([47cd7d3](https://github.com/UCSD-E4E/fishsense-core/commit/47cd7d3ec20c16aaaa6fdcdad2240e76cc113fed))
* readd missing files ([f797879](https://github.com/UCSD-E4E/fishsense-core/commit/f797879a31115642eb1840cd1f7ae47da4e7a23f))
* release please releases should be linked ([2760e21](https://github.com/UCSD-E4E/fishsense-core/commit/2760e21747d4d27f23d0bd002138bbacdf2a411c))
* specify native module ([c7589c8](https://github.com/UCSD-E4E/fishsense-core/commit/c7589c8520ade8ec9ee82cfb658b4e4decfdb495))
* split the module ([e8eeb1d](https://github.com/UCSD-E4E/fishsense-core/commit/e8eeb1d2aa9c2269d0637cf35fbfbbcb01657741))
* try changing the module name ([e0a4f1e](https://github.com/UCSD-E4E/fishsense-core/commit/e0a4f1e1a94e5d1aaffe65aac048fdc6b2731705))
* try to export calibrate laser again ([19844ee](https://github.com/UCSD-E4E/fishsense-core/commit/19844eeb4a6412b842007331bfc1e909904d526e))
* try to use a different name for module ([e4c33fd](https://github.com/UCSD-E4E/fishsense-core/commit/e4c33fd52e6127183967960f37e0ca6f22bc8c14))

## [0.3.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense_core-v0.2.0...fishsense_core-v0.3.0) (2026-04-12)


### Features

* additional logging.  gpu-based connected compoennts implementation ([cc18072](https://github.com/UCSD-E4E/fishsense-core/commit/cc18072cb54a12aebb6656f6b232f72c87156f26))

## [0.2.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense_core-v0.1.0...fishsense_core-v0.2.0) (2026-04-11)


### Features

* add base image file from previous implementation ([57d5a81](https://github.com/UCSD-E4E/fishsense-core/commit/57d5a816e38b028b8b114d8a0b0aefb999604ab4))
* add missing raw image and rectified image ([7ca7800](https://github.com/UCSD-E4E/fishsense-core/commit/7ca78007e9a08dcdc92703af88a06a7116ab2f97))
* expose calibrate laser method ([1a7d903](https://github.com/UCSD-E4E/fishsense-core/commit/1a7d903463944b1af2ec76dc05e775340fb74336))
* initial checkin ([916aa0b](https://github.com/UCSD-E4E/fishsense-core/commit/916aa0b92c99dd9331b124ff753879cf852a2154))
* let's try to properly export laser ([e6aebf8](https://github.com/UCSD-E4E/fishsense-core/commit/e6aebf84080c13e7d50ea000aec2ad5b1e4359bf))
* release please ([2b752d4](https://github.com/UCSD-E4E/fishsense-core/commit/2b752d4e7f7822230f143c1452afc32bb2666625))
* release structure and readme ([dfe694f](https://github.com/UCSD-E4E/fishsense-core/commit/dfe694feb22f4aca44cdf45c4d8298166e4e6c44))
* switch fishsense_core python package to maturin ([8d8b7cb](https://github.com/UCSD-E4E/fishsense-core/commit/8d8b7cb7997dbbd109b1ed24b11117ccad0062ff))


### Bug Fixes

* add missing module file ([1eb64d7](https://github.com/UCSD-E4E/fishsense-core/commit/1eb64d748a7976b36ab10d31d08fad4fb2db58cf))
* do calibration in f64 rather than f32 ([24fd90d](https://github.com/UCSD-E4E/fishsense-core/commit/24fd90dfa25af25e88da0f989c1cf5c233b69e9f))
* ensure tha the laser_origin z is on the z=0 plane ([0bd7d72](https://github.com/UCSD-E4E/fishsense-core/commit/0bd7d72c531b2ae16bfcc02d23d85302eb57c0e2))
* export the native module ([2ae0514](https://github.com/UCSD-E4E/fishsense-core/commit/2ae05147b42f947c5505eb15b278677c9a6a5e6a))
* let's try a different module syntax ([d323564](https://github.com/UCSD-E4E/fishsense-core/commit/d3235644d7322cea7b93debc919d6fd7e53fd14a))
* let's try laser module again ([9932a20](https://github.com/UCSD-E4E/fishsense-core/commit/9932a2064ecf8fbfd7864c42044e582decc1cc23))
* let's try updaing the module paths again ([7856fc8](https://github.com/UCSD-E4E/fishsense-core/commit/7856fc872dd437dcb8befed916ba7c48ab2f6aad))
* pylint errors ([71f5c87](https://github.com/UCSD-E4E/fishsense-core/commit/71f5c8743a96caad9b761c005dd698b59fa260bd))
* python dependencies ([6bf2e1c](https://github.com/UCSD-E4E/fishsense-core/commit/6bf2e1c2aceaaa8bc2a7fbacafe0b2c8fab130e7))
* python tests now pass ([47cd7d3](https://github.com/UCSD-E4E/fishsense-core/commit/47cd7d3ec20c16aaaa6fdcdad2240e76cc113fed))
* readd missing files ([f797879](https://github.com/UCSD-E4E/fishsense-core/commit/f797879a31115642eb1840cd1f7ae47da4e7a23f))
* specify native module ([c7589c8](https://github.com/UCSD-E4E/fishsense-core/commit/c7589c8520ade8ec9ee82cfb658b4e4decfdb495))
* split the module ([e8eeb1d](https://github.com/UCSD-E4E/fishsense-core/commit/e8eeb1d2aa9c2269d0637cf35fbfbbcb01657741))
* try changing the module name ([e0a4f1e](https://github.com/UCSD-E4E/fishsense-core/commit/e0a4f1e1a94e5d1aaffe65aac048fdc6b2731705))
* try to export calibrate laser again ([19844ee](https://github.com/UCSD-E4E/fishsense-core/commit/19844eeb4a6412b842007331bfc1e909904d526e))
* try to use a different name for module ([e4c33fd](https://github.com/UCSD-E4E/fishsense-core/commit/e4c33fd52e6127183967960f37e0ca6f22bc8c14))
