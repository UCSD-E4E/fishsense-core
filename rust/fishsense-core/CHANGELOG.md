# Changelog

## [2.1.5](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v2.1.4...fishsense-core-v2.1.5) (2026-07-03)


### Bug Fixes

* stabilize fishial.onnx download in build.rs ([596fd54](https://github.com/UCSD-E4E/fishsense-core/commit/596fd54252125f97a5b5e5e156289acdcd3fb2ee))

## [2.1.4](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v2.1.3...fishsense-core-v2.1.4) (2026-05-13)


### Bug Fixes

* silence ONNX Runtime stderr spam on the no-fish segmentation path ([e435083](https://github.com/UCSD-E4E/fishsense-core/commit/e435083c9b72785e60d3793820d349ae061a6ab3))
* silence ONNX Runtime stderr spam on the no-fish segmentation path ([2ec65a5](https://github.com/UCSD-E4E/fishsense-core/commit/2ec65a5ed5f6c04d69e67309e75e65babda70806))

## [2.1.3](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v2.1.2...fishsense-core-v2.1.3) (2026-05-11)


### Bug Fixes

* build the CUDA wheel against onnxruntime-gpu so it loads on glibc &lt; 2.38 ([dbfc38e](https://github.com/UCSD-E4E/fishsense-core/commit/dbfc38e3ba19dfbc396fc31aa1d1a410d87cd14d))
* build the CUDA wheel against onnxruntime-gpu so it loads on glibc &lt; 2.38 ([939b292](https://github.com/UCSD-E4E/fishsense-core/commit/939b292935091c3098961e8bcafc4705ee29f8f9))

## [2.1.2](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v2.1.1...fishsense-core-v2.1.2) (2026-05-10)


### Bug Fixes

* cap ORT intra-op threads at CPU count; add CI faulthandler watchdog ([aed4f41](https://github.com/UCSD-E4E/fishsense-core/commit/aed4f41b55bd6b7a8ac4d9623a57cd6fb2a8e159))
* cap ORT intra-op threads at CPU count; add CI faulthandler watchdog ([efe6c53](https://github.com/UCSD-E4E/fishsense-core/commit/efe6c53468de2c4a1e190e3bde7e37b1332e86bb))
* make ORT intra-op thread count overridable; run sessions single-threaded in CI ([3d90563](https://github.com/UCSD-E4E/fishsense-core/commit/3d905639987d7952838dea36df2ee0ac4574d5a4))
* ORT intra-op threads overridable via env; run sessions single-threaded in CI ([1a9c5da](https://github.com/UCSD-E4E/fishsense-core/commit/1a9c5da0a3ed50f0cd61e16d83c41e6a6d80a431))

## [2.1.1](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v2.1.0...fishsense-core-v2.1.1) (2026-05-09)


### Bug Fixes

* enable openblas-src rustls feature so openblas-build 0.10.16 compiles ([798e70e](https://github.com/UCSD-E4E/fishsense-core/commit/798e70ee544a21e140021e63bdb52c547424f88e))
* enable openblas-src rustls feature so openblas-build 0.10.16 compiles ([fa7da32](https://github.com/UCSD-E4E/fishsense-core/commit/fa7da32d27f2fd66502e4b4603cdea22e1fd02e6))
* preload nvidia CUDA libs and expose active EP ([932c9d6](https://github.com/UCSD-E4E/fishsense-core/commit/932c9d6e01df3c2c5494abfedcb933aa74964013))
* preload nvidia CUDA libs and expose active execution provider ([eb0f0de](https://github.com/UCSD-E4E/fishsense-core/commit/eb0f0deaa28c42709c4f5df290b4765497feaf51))

## [2.1.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v2.0.0...fishsense-core-v2.1.0) (2026-05-06)


### Features

* add CUDA execution provider for fish segmentation ([fe7724f](https://github.com/UCSD-E4E/fishsense-core/commit/fe7724feac85985f87adeebd923db33ae5a3f47f))
* add CUDA execution provider for fish segmentation ([80229f5](https://github.com/UCSD-E4E/fishsense-core/commit/80229f5a1d1b7bdbb8dacea3f1785814bd974f20))
* auto-stage ORT CUDA provider libs into the python wheel ([7cb5ea6](https://github.com/UCSD-E4E/fishsense-core/commit/7cb5ea60b5bc5e3452e62fa37fa9dc1d0ab96f37))

## [2.0.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v1.7.1...fishsense-core-v2.0.0) (2026-04-29)


### ⚠ BREAKING CHANGES

* `FishHeadTailDetector::snap_to_depth_map` and `find_head_tail_depth` are gone; callers needing per-keypoint depth should use `predict_keypoint_depths(depth_map, mask, k_inv, snout_xy, fork_xy)` instead. `spatial::connected_components`, `gpu`, and

### Features

* replace depth-snap with mask-bounded RANSAC plane-fit ([efdfbfa](https://github.com/UCSD-E4E/fishsense-core/commit/efdfbfaa32b72a45719995954d940d2d5fb14632))

## [1.7.1](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v1.7.0...fishsense-core-v1.7.1) (2026-04-28)


### Miscellaneous Chores

* **fishsense-core:** Synchronize fishsense versions

## [1.7.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v1.6.0...fishsense-core-v1.7.0) (2026-04-27)


### Miscellaneous Chores

* **fishsense-core:** Synchronize fishsense versions

## [1.6.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v1.5.0...fishsense-core-v1.6.0) (2026-04-27)


### Features

* bind WorldPointHandler to Python with project + laser triangulation ([ce5a863](https://github.com/UCSD-E4E/fishsense-core/commit/ce5a863039baf8c0dbc92f1d67cddc39d5ea76c2))


### Bug Fixes

* drop post-snap mask-overlap fallback in find_head_tail_depth ([fa47ba6](https://github.com/UCSD-E4E/fishsense-core/commit/fa47ba6852aee91e89b21a74b6610847760e4869))
* guard find_head_tail_depth against depth-discontinuity collapse ([1cd5f99](https://github.com/UCSD-E4E/fishsense-core/commit/1cd5f991c39ff2c3e18e7da3aa562c45eb797c41))

## [1.5.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v1.4.1...fishsense-core-v1.5.0) (2026-04-26)


### Features

* return mask-space coords from find_head_tail_depth ([08577b3](https://github.com/UCSD-E4E/fishsense-core/commit/08577b3640b9cb640ce91756655864248af1a3ca))

## [1.4.1](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v1.4.0...fishsense-core-v1.4.1) (2026-04-26)


### Bug Fixes

* rescale head/tail coords into depth grid before snapping ([0f40af0](https://github.com/UCSD-E4E/fishsense-core/commit/0f40af0e2861ccb9d85b4f5fbe204c0909bf144f))

## [1.4.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v1.3.0...fishsense-core-v1.4.0) (2026-04-21)


### Features

* add coreml feature and gate optimization level for iOS ([10680ef](https://github.com/UCSD-E4E/fishsense-core/commit/10680efeadd34bfac6433eb10d7668c1d317fede))

## [1.3.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v1.2.0...fishsense-core-v1.3.0) (2026-04-21)


### Features

* add inference_single for single-fish mask output ([ea196dd](https://github.com/UCSD-E4E/fishsense-core/commit/ea196ddeedb9c9f8a7f000a1efd35bc0a20fed79))


### Bug Fixes

* classify head/tail by concavity proximity, not total area ([c5b686c](https://github.com/UCSD-E4E/fishsense-core/commit/c5b686c55f29ff551c45c377f716b1fd80884fe3))
* head/tail endpoint snap and boundary-min orientation ([a287305](https://github.com/UCSD-E4E/fishsense-core/commit/a2873051328299d5b119c49a16b6911ba63339f4))
* head/tail orientation via peduncle + hull-area cascade ([65731fb](https://github.com/UCSD-E4E/fishsense-core/commit/65731fb1008064ac7fd01848c76d6d53439e2669))
* pick nearest significant concavity, not largest, for head/tail ([a935d56](https://github.com/UCSD-E4E/fishsense-core/commit/a935d56191dd2d357eec61f09148ff13c2c74b1a))

## [1.2.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v1.1.0...fishsense-core-v1.2.0) (2026-04-20)


### Bug Fixes

* head/tail refinement fails ([e4c4dc7](https://github.com/UCSD-E4E/fishsense-core/commit/e4c4dc722ddd66a5f5196653c651faaceeb5bf0e))

## [1.1.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v1.0.0...fishsense-core-v1.1.0) (2026-04-20)


### Miscellaneous Chores

* **fishsense-core:** Synchronize fishsense versions

## [1.0.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v0.8.0...fishsense-core-v1.0.0) (2026-04-14)


### ⚠ BREAKING CHANGES

* correct depth indexing convention and support downsampled depth maps

### Bug Fixes

* correct depth indexing convention and support downsampled depth maps ([c1dfdc3](https://github.com/UCSD-E4E/fishsense-core/commit/c1dfdc32b938bca59052067685f97ddc603a6755))

## [0.8.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v0.7.0...fishsense-core-v0.8.0) (2026-04-14)


### Features

* add additional tracing ([321b5a2](https://github.com/UCSD-E4E/fishsense-core/commit/321b5a210d778671481a01dde353b8d690d4cf07))
* add fish length calculation and world coord handling ([2e838c7](https://github.com/UCSD-E4E/fishsense-core/commit/2e838c75b583e9fae21cedef4b0e5a97ca4d7edb))
* additional logging.  gpu-based connected compoennts implementation ([cc18072](https://github.com/UCSD-E4E/fishsense-core/commit/cc18072cb54a12aebb6656f6b232f72c87156f26))
* bring all versions up to date ([2ed4c25](https://github.com/UCSD-E4E/fishsense-core/commit/2ed4c25727f0f7c6982a474b949056056357c4a8))
* cpu fallback for connected components. ([345073f](https://github.com/UCSD-E4E/fishsense-core/commit/345073f88ea803b18465d6d7e385e52d827b7021))
* expose calibrate laser method ([1a7d903](https://github.com/UCSD-E4E/fishsense-core/commit/1a7d903463944b1af2ec76dc05e775340fb74336))
* fish head tail snap to depth map ([50120bb](https://github.com/UCSD-E4E/fishsense-core/commit/50120bbac317e82385296ecfbb625a5118f650ae))
* fishial-based segmentation ([2476059](https://github.com/UCSD-E4E/fishsense-core/commit/2476059cd1a6957874b7bd505d0f3b6894b0221d))
* introduce fishsense-core rust package ([6dde82c](https://github.com/UCSD-E4E/fishsense-core/commit/6dde82c152ca99d621b9955bcb744258aed20a42))
* introduce head/tail detector ([5f902d4](https://github.com/UCSD-E4E/fishsense-core/commit/5f902d4f53f2d445cd97668ae6fd74c3c6f77a89))
* rust implementation of laser calibration ([f5675ea](https://github.com/UCSD-E4E/fishsense-core/commit/f5675ea352932613d434aca556eafb664ea02965))
* unit tests.  run pytest on ubuntu ([27b64b7](https://github.com/UCSD-E4E/fishsense-core/commit/27b64b725f6cb26289e0df855d24f7c679b12924))


### Bug Fixes

* add missing export ([ef213bf](https://github.com/UCSD-E4E/fishsense-core/commit/ef213bf434452d9290ee24feb1d0159d03e1e4fc))
* build downloads fishial from onnx.  need to use a connection timeout not a stream timeout ([0ce6d0b](https://github.com/UCSD-E4E/fishsense-core/commit/0ce6d0bc0944adffc0ae4e348fbe5ace721d165c))
* clippy issues ([79e86f7](https://github.com/UCSD-E4E/fishsense-core/commit/79e86f7cebc1b33dc7634664d0ab88a8785baa42))
* do calibration in f64 rather than f32 ([24fd90d](https://github.com/UCSD-E4E/fishsense-core/commit/24fd90dfa25af25e88da0f989c1cf5c233b69e9f))
* ensure tha the laser_origin z is on the z=0 plane ([0bd7d72](https://github.com/UCSD-E4E/fishsense-core/commit/0bd7d72c531b2ae16bfcc02d23d85302eb57c0e2))
* fishsense_core package name ([154514b](https://github.com/UCSD-E4E/fishsense-core/commit/154514b3a3a45124170b8d13e1e700a494d2c274))
* prevent OOM in extract_perimeter on degenerate masks ([47e284a](https://github.com/UCSD-E4E/fishsense-core/commit/47e284a40ec378d877fa7f6256dd71ffcd03a156))
* print centroid option ([7fd7442](https://github.com/UCSD-E4E/fishsense-core/commit/7fd74426aab94b166836f553d7db188c0d046729))
* rust build issues and release please build issues ([47061b9](https://github.com/UCSD-E4E/fishsense-core/commit/47061b9b59f324d9c07f617912ee4855cc6c946e))
* update opencv version and make sure that cargo can find libclang ([62cfe1d](https://github.com/UCSD-E4E/fishsense-core/commit/62cfe1d3b6d16cc8f54f7cfeb1a8ed9c7f77eb61))

## [0.4.2](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v0.4.1...fishsense-core-v0.4.2) (2026-04-14)


### Bug Fixes

* prevent OOM in extract_perimeter on degenerate masks ([47e284a](https://github.com/UCSD-E4E/fishsense-core/commit/47e284a40ec378d877fa7f6256dd71ffcd03a156))

## [0.4.1](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v0.4.0...fishsense-core-v0.4.1) (2026-04-13)


### Bug Fixes

* build downloads fishial from onnx.  need to use a connection timeout not a stream timeout ([0ce6d0b](https://github.com/UCSD-E4E/fishsense-core/commit/0ce6d0bc0944adffc0ae4e348fbe5ace721d165c))

## [0.4.0](https://github.com/UCSD-E4E/fishsense-core/compare/fishsense-core-v0.3.0...fishsense-core-v0.4.0) (2026-04-13)


### Features

* add fish length calculation and world coord handling ([2e838c7](https://github.com/UCSD-E4E/fishsense-core/commit/2e838c75b583e9fae21cedef4b0e5a97ca4d7edb))


### Bug Fixes

* clippy issues ([79e86f7](https://github.com/UCSD-E4E/fishsense-core/commit/79e86f7cebc1b33dc7634664d0ab88a8785baa42))

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
