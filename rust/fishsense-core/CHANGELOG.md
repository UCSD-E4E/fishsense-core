# Changelog

## [0.3.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v0.2.0...fishsense-core-v0.3.0) (2026-04-12)


### Features

* add additional tracing ([321b5a2](https://github.com/UCSD-E4E/fishsense-core/commit/321b5a210d778671481a01dde353b8d690d4cf07))
* additional logging.  gpu-based connected compoennts implementation ([cc18072](https://github.com/UCSD-E4E/fishsense-core/commit/cc18072cb54a12aebb6656f6b232f72c87156f26))
* cpu fallback for connected components. ([345073f](https://github.com/UCSD-E4E/fishsense-core/commit/345073f88ea803b18465d6d7e385e52d827b7021))
* fish head tail snap to depth map ([50120bb](https://github.com/UCSD-E4E/fishsense-core/commit/50120bbac317e82385296ecfbb625a5118f650ae))
* fishial-based segmentation ([2476059](https://github.com/UCSD-E4E/fishsense-core/commit/2476059cd1a6957874b7bd505d0f3b6894b0221d))
* introduce head/tail detector ([5f902d4](https://github.com/UCSD-E4E/fishsense-core/commit/5f902d4f53f2d445cd97668ae6fd74c3c6f77a89))


### Bug Fixes

* rust build issues and release please build issues ([47061b9](https://github.com/UCSD-E4E/fishsense-core/commit/47061b9b59f324d9c07f617912ee4855cc6c946e))
* update opencv version and make sure that cargo can find libclang ([62cfe1d](https://github.com/UCSD-E4E/fishsense-core/commit/62cfe1d3b6d16cc8f54f7cfeb1a8ed9c7f77eb61))

## [0.2.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v0.1.0...fishsense-core-v0.2.0) (2026-04-11)


### Features

* expose calibrate laser method ([1a7d903](https://github.com/UCSD-E4E/fishsense-core/commit/1a7d903463944b1af2ec76dc05e775340fb74336))
* introduce fishsense-core rust package ([6dde82c](https://github.com/UCSD-E4E/fishsense-core/commit/6dde82c152ca99d621b9955bcb744258aed20a42))
* rust implementation of laser calibration ([f5675ea](https://github.com/UCSD-E4E/fishsense-core/commit/f5675ea352932613d434aca556eafb664ea02965))
* unit tests.  run pytest on ubuntu ([27b64b7](https://github.com/UCSD-E4E/fishsense-core/commit/27b64b725f6cb26289e0df855d24f7c679b12924))


### Bug Fixes

* add missing export ([ef213bf](https://github.com/UCSD-E4E/fishsense-core/commit/ef213bf434452d9290ee24feb1d0159d03e1e4fc))
* do calibration in f64 rather than f32 ([24fd90d](https://github.com/UCSD-E4E/fishsense-core/commit/24fd90dfa25af25e88da0f989c1cf5c233b69e9f))
* ensure tha the laser_origin z is on the z=0 plane ([0bd7d72](https://github.com/UCSD-E4E/fishsense-core/commit/0bd7d72c531b2ae16bfcc02d23d85302eb57c0e2))
* fishsense_core package name ([154514b](https://github.com/UCSD-E4E/fishsense-core/commit/154514b3a3a45124170b8d13e1e700a494d2c274))
* print centroid option ([7fd7442](https://github.com/UCSD-E4E/fishsense-core/commit/7fd74426aab94b166836f553d7db188c0d046729))
