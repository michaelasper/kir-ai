# Changelog

All notable changes to this project will be documented in this file.

## [1.25.0](https://github.com/michaelasper/kir-ai/compare/v1.24.2...v1.25.0) (2026-05-21)

### Features

* **COR-336:** rate limit public inference endpoints ([9d88555](https://github.com/michaelasper/kir-ai/commit/9d88555b71bbd3c6391243493891cc8ba9aee87e))

## [1.24.2](https://github.com/michaelasper/kir-ai/compare/v1.24.1...v1.24.2) (2026-05-21)

### Bug Fixes

* **COR-335:** sanitize streaming SSE errors ([8e5be73](https://github.com/michaelasper/kir-ai/commit/8e5be73d71feb1faa4564d17eb234fcdf2c5d57f))

## [1.24.1](https://github.com/michaelasper/kir-ai/compare/v1.24.0...v1.24.1) (2026-05-21)

### Bug Fixes

* **COR-290:** count schema validation tool failures ([3eb1682](https://github.com/michaelasper/kir-ai/commit/3eb168253000c7c592f7bcf2cd3316bc8df62d5e))

## [1.24.0](https://github.com/michaelasper/kir-ai/compare/v1.23.0...v1.24.0) (2026-05-21)

### Features

* **COR-291:** centralize no-progress thresholds ([7bf66bc](https://github.com/michaelasper/kir-ai/commit/7bf66bc6be0ea9468c1349c140542daab088f012))

## [1.23.0](https://github.com/michaelasper/kir-ai/compare/v1.22.2...v1.23.0) (2026-05-21)

### Features

* **COR-237:** add snapshot readiness modes ([ba7eb13](https://github.com/michaelasper/kir-ai/commit/ba7eb133905ace5a724db63fd5ebce210dd77d08))

## [1.22.2](https://github.com/michaelasper/kir-ai/compare/v1.22.1...v1.22.2) (2026-05-21)

### Bug Fixes

* **COR-236:** avoid double snapshot verification ([62530cd](https://github.com/michaelasper/kir-ai/commit/62530cd7332bbd3629273dd316b19f891e0967a8))

## [1.22.1](https://github.com/michaelasper/kir-ai/compare/v1.22.0...v1.22.1) (2026-05-20)

### Bug Fixes

* **COR-233:** standardize native max token resolution ([ae90454](https://github.com/michaelasper/kir-ai/commit/ae90454186c1d89da1694b12db2d56ebc12d7688))

## [1.22.0](https://github.com/michaelasper/kir-ai/compare/v1.21.0...v1.22.0) (2026-05-20)

### Features

* **COR-232:** centralize cache byte accounting ([62ff7dd](https://github.com/michaelasper/kir-ai/commit/62ff7dd1de2bd895ab5c03b405c1bb2f26b92921))

## [1.21.0](https://github.com/michaelasper/kir-ai/compare/v1.20.0...v1.21.0) (2026-05-20)

### Features

* **COR-231:** add warm prefix benchmark cases ([4a45f2b](https://github.com/michaelasper/kir-ai/commit/4a45f2b85c9959911644b987df86833ade2198bc))

## [1.20.0](https://github.com/michaelasper/kir-ai/compare/v1.19.0...v1.20.0) (2026-05-20)

### Features

* **COR-230:** add native prefill work metrics ([6c046e2](https://github.com/michaelasper/kir-ai/commit/6c046e2cf047df4544b7d88b3d3eadcec6c929be))

## [1.19.0](https://github.com/michaelasper/kir-ai/compare/v1.18.0...v1.19.0) (2026-05-20)

### Features

* **COR-229:** expose native prefix cache byte limit ([e835904](https://github.com/michaelasper/kir-ai/commit/e835904d3a414790faa233a469fbf87f5c3a44b8))

## [1.18.0](https://github.com/michaelasper/kir-ai/compare/v1.17.10...v1.18.0) (2026-05-20)

### Features

* **COR-228:** expose prefix cache lookup and clone metrics ([6784700](https://github.com/michaelasper/kir-ai/commit/6784700fd4f19cb862e8e0fce49f80f0ceafe90d))

## [1.17.10](https://github.com/michaelasper/kir-ai/compare/v1.17.9...v1.17.10) (2026-05-20)

### Bug Fixes

* **COR-220:** harden request validation before rendering ([a22e1f3](https://github.com/michaelasper/kir-ai/commit/a22e1f3aed4fe4996eac72466021e7e783953567))

## [1.17.9](https://github.com/michaelasper/kir-ai/compare/v1.17.8...v1.17.9) (2026-05-20)

### Bug Fixes

* **COR-219:** stop using cache schema as tool transport ([237cfde](https://github.com/michaelasper/kir-ai/commit/237cfde45c90cc649af1ba07e365873287b0874d))

## [1.17.8](https://github.com/michaelasper/kir-ai/compare/v1.17.7...v1.17.8) (2026-05-20)

### Bug Fixes

* **COR-216:** bind validated requests to limits ([bb7eb01](https://github.com/michaelasper/kir-ai/commit/bb7eb01b0be1141c69efd1a518ed91ed079ef891))
* **COR-216:** pass validated requests into runtime ([a7a25fa](https://github.com/michaelasper/kir-ai/commit/a7a25fa055fccc06aa8cecd3ba3c4daee742d337))

## [1.17.7](https://github.com/michaelasper/kir-ai/compare/v1.17.6...v1.17.7) (2026-05-20)

### Bug Fixes

* **COR-205:** remove stale native qwen cfg-test wrappers ([74b29c7](https://github.com/michaelasper/kir-ai/commit/74b29c773c893f4824826e1187ede15e28202a62))

## [1.17.6](https://github.com/michaelasper/kir-ai/compare/v1.17.5...v1.17.6) (2026-05-20)

### Bug Fixes

* **COR-204:** finish qwen moe and matvec paths ([dbcb93d](https://github.com/michaelasper/kir-ai/commit/dbcb93d07bb4c19570aa00e952c6157c4ebc52e1))

## [1.17.5](https://github.com/michaelasper/kir-ai/compare/v1.17.4...v1.17.5) (2026-05-20)

### Bug Fixes

* **COR-203:** normalize native text next-token adapter ([cc59e0f](https://github.com/michaelasper/kir-ai/commit/cc59e0f16adde654814b5b957bb46fd5bef24224))

## [1.17.4](https://github.com/michaelasper/kir-ai/compare/v1.17.3...v1.17.4) (2026-05-20)

### Bug Fixes

* **COR-201:** restore workspace all-targets compile ([88032a0](https://github.com/michaelasper/kir-ai/commit/88032a0ceb8eda05d898877f4ab6fb2b86e8c641))

## [1.17.3](https://github.com/michaelasper/kir-ai/compare/v1.17.2...v1.17.3) (2026-05-16)

### Bug Fixes

* **admin:** serialize model pull requests ([6b0982b](https://github.com/michaelasper/kir-ai/commit/6b0982b807e6029d08421f47a28fb5ba65f357a4)), closes [#162](https://github.com/michaelasper/kir-ai/issues/162)

## [1.17.2](https://github.com/michaelasper/kir-ai/compare/v1.17.1...v1.17.2) (2026-05-16)

### Bug Fixes

* **runtime:** bound repeated invalid tool calls ([52cccd8](https://github.com/michaelasper/kir-ai/commit/52cccd83cffc51e44e0f4d3b4ecaaf34a9b28fb5))
* **streaming:** track hidden chat progress for stall deadlines ([75a058d](https://github.com/michaelasper/kir-ai/commit/75a058d948e91f715255fa60ba966d1407c6a1d1))

## [1.17.1](https://github.com/michaelasper/kir-ai/compare/v1.17.0...v1.17.1) (2026-05-16)

### Bug Fixes

* **mlx:** preserve non-stream Qwen XML structured tool calls ([12961ab](https://github.com/michaelasper/kir-ai/commit/12961ab1c203dec2aea40cb5cd692a480fae3173))

## [1.17.0](https://github.com/michaelasper/kir-ai/compare/v1.16.0...v1.17.0) (2026-05-16)

### Features

* **kv-cache:** add restorable native cache snapshots ([a8840a8](https://github.com/michaelasper/kir-ai/commit/a8840a8de710d1455b9965cdbac6698da77447b3))

## [1.16.0](https://github.com/michaelasper/kir-ai/compare/v1.15.3...v1.16.0) (2026-05-15)

### Features

* **native:** stream prefill progress events ([e081a77](https://github.com/michaelasper/kir-ai/commit/e081a771a0df47ff736b45fea6c3eb636497204f))

## [1.15.3](https://github.com/michaelasper/kir-ai/compare/v1.15.2...v1.15.3) (2026-05-15)

### Performance Improvements

* **native:** store metal kv mirrors as f16 ([bb1e3f2](https://github.com/michaelasper/kir-ai/commit/bb1e3f2453dc22868ac3853f2c6e20c2e1f0a772))

## [1.15.2](https://github.com/michaelasper/kir-ai/compare/v1.15.1...v1.15.2) (2026-05-15)

### Performance Improvements

* **native:** grow metal kv mirrors by token blocks ([63212ce](https://github.com/michaelasper/kir-ai/commit/63212ce356068501d2d40e4e2807308cee9e4fc4))

## [1.15.1](https://github.com/michaelasper/kir-ai/compare/v1.15.0...v1.15.1) (2026-05-15)

### Bug Fixes

* **native:** incrementally sync metal kv cache appends ([0f9aea8](https://github.com/michaelasper/kir-ai/commit/0f9aea8af2f98cd4756ef7495ec54935652cde35))

## [1.15.0](https://github.com/michaelasper/kir-ai/compare/v1.14.7...v1.15.0) (2026-05-15)

### Features

* **mlx:** surface zero-output successes ([090cacb](https://github.com/michaelasper/kir-ai/commit/090cacb81c7aa895c16815599196cf9013b255bf))

## [1.14.7](https://github.com/michaelasper/kir-ai/compare/v1.14.6...v1.14.7) (2026-05-15)

### Bug Fixes

* **mlx:** allow long prefill before stream bytes ([a16c354](https://github.com/michaelasper/kir-ai/commit/a16c354fe324b9df0b26367faffb91c312cb4384))

## [1.14.6](https://github.com/michaelasper/kir-ai/compare/v1.14.5...v1.14.6) (2026-05-15)

### Bug Fixes

* **mlx:** tolerate length-truncated qwen xml tools ([3e7863d](https://github.com/michaelasper/kir-ai/commit/3e7863d07d674ca5887118b086b2581e661ef9b5))

## [1.14.5](https://github.com/michaelasper/kir-ai/compare/v1.14.4...v1.14.5) (2026-05-15)

### Bug Fixes

* **native:** keep prefix cache identity bucketed ([2751842](https://github.com/michaelasper/kir-ai/commit/27518426cf29f19546ce88b1daa34b3e1223dfa2))

## [1.14.4](https://github.com/michaelasper/kir-ai/compare/v1.14.3...v1.14.4) (2026-05-15)

### Bug Fixes

* **bench:** remove unsupported gemma vlm max tokens flag ([9babb42](https://github.com/michaelasper/kir-ai/commit/9babb42fd15a7d12816c2784831a3db1cd54f72d))

### Performance Improvements

* **native:** sync only live kv cache tokens ([ef3dc9f](https://github.com/michaelasper/kir-ai/commit/ef3dc9f9d22e05c783341e2817d2d0754c9f9eb1))

## [1.14.3](https://github.com/michaelasper/kir-ai/compare/v1.14.2...v1.14.3) (2026-05-15)

### Bug Fixes

* **mlx:** tolerate qwen streaming parser edge cases ([b045882](https://github.com/michaelasper/kir-ai/commit/b0458824e84aa3c46c1815c463a9e728d5f88736))

## [1.14.2](https://github.com/michaelasper/kir-ai/compare/v1.14.1...v1.14.2) (2026-05-15)

### Bug Fixes

* **native:** raise prefill chunk default ([12767c6](https://github.com/michaelasper/kir-ai/commit/12767c6f66f9855281dff267bc4e10a9cd402de5))

## [1.14.1](https://github.com/michaelasper/kir-ai/compare/v1.14.0...v1.14.1) (2026-05-15)

### Bug Fixes

* **api:** allow configurable long-context request limits ([b787203](https://github.com/michaelasper/kir-ai/commit/b787203d48d996c5877dc44452702ea958c392cc))
* **gemma:** propagate no-thinking template identity ([ccafa4a](https://github.com/michaelasper/kir-ai/commit/ccafa4a70ec6be8269d8fe1b590154207accaa8b))
* **hub:** prefer LFS sha256 over blob id ([d9c4aca](https://github.com/michaelasper/kir-ai/commit/d9c4acaa29bd27430d1f5ea93de59b2b17756ccf))
* **streaming:** start stall timer after first output ([2ca8e55](https://github.com/michaelasper/kir-ai/commit/2ca8e55afb13214472ce03a3d6dbde2de9b99e07))

## [1.14.0](https://github.com/michaelasper/kir-ai/compare/v1.13.0...v1.14.0) (2026-05-15)

### Features

* **engine:** gate backend implementations ([9e57d06](https://github.com/michaelasper/kir-ai/commit/9e57d06fbd3a9781b1a8c08f2636823949efaf24))
* **engine:** gate diagnostics commands ([b2af317](https://github.com/michaelasper/kir-ai/commit/b2af3171c2a3ef04f33b469abd7e7a128d913313))
* **server:** map backend metrics by provider ([22b6e54](https://github.com/michaelasper/kir-ai/commit/22b6e54f4465a647a59a7e83700b420ee9b00de1))
* **telemetry:** gate tool-call metrics ([2768416](https://github.com/michaelasper/kir-ai/commit/2768416d6cec4fca303361970fd329487b44b813))

### Bug Fixes

* **server:** derive backend metrics defaults ([dda7c06](https://github.com/michaelasper/kir-ai/commit/dda7c0628a7121efc47ddbb713b546fd2ea8368b))

## [1.13.0](https://github.com/michaelasper/kir-ai/compare/v1.12.0...v1.13.0) (2026-05-14)

### Features

* **cache:** add stable prefix metrics ([c79a7e9](https://github.com/michaelasper/kir-ai/commit/c79a7e9f3cb3b463f92a4a795ae0634e35752e26))

## [1.12.0](https://github.com/michaelasper/kir-ai/compare/v1.11.0...v1.12.0) (2026-05-14)

### Features

* **bench:** add qwen mlx prefill sweep ([e7caa08](https://github.com/michaelasper/kir-ai/commit/e7caa08a79a4dbdb3878f5999604c802a9f5c37e)), closes [#263](https://github.com/michaelasper/kir-ai/issues/263)

## [1.11.0](https://github.com/michaelasper/kir-ai/compare/v1.10.3...v1.11.0) (2026-05-13)

### Features

* add tool stream timing telemetry ([5d5026f](https://github.com/michaelasper/kir-ai/commit/5d5026fd789d4d54cc91f3577baa023c72003a56)), closes [#261](https://github.com/michaelasper/kir-ai/issues/261)

## [1.10.3](https://github.com/michaelasper/kir-ai/compare/v1.10.2...v1.10.3) (2026-05-13)

### Bug Fixes

* **mlx:** forward streaming usage ([41061c5](https://github.com/michaelasper/kir-ai/commit/41061c56d3dee14bc8066e0d5cd66d3996b7dd13))

## [1.10.2](https://github.com/michaelasper/kir-ai/compare/v1.10.1...v1.10.2) (2026-05-13)

### Bug Fixes

* **backend:** avoid rms norm zero-scale NaNs ([9b350a4](https://github.com/michaelasper/kir-ai/commit/9b350a4a3d0d95b8f8f832330ad7d7188a226221))

## [1.10.1](https://github.com/michaelasper/kir-ai/compare/v1.10.0...v1.10.1) (2026-05-13)

### Bug Fixes

* **scheduler:** classify by token estimate ([65a4644](https://github.com/michaelasper/kir-ai/commit/65a464407ad8b46359a4ff41a97fde36c64fdb85))

## [1.10.0](https://github.com/michaelasper/kir-ai/compare/v1.9.0...v1.10.0) (2026-05-13)

### Features

* **bench:** add structured qwen report metrics ([7f20d29](https://github.com/michaelasper/kir-ai/commit/7f20d29ef6d527dfc8e5fe70eee4f9a13631daed))

## [1.9.0](https://github.com/michaelasper/kir-ai/compare/v1.8.12...v1.9.0) (2026-05-13)

### Features

* **native:** expose prefix cache reuse per request ([ed0aeaf](https://github.com/michaelasper/kir-ai/commit/ed0aeafec705028e96fd6407196fd03851be1137))

## [1.8.12](https://github.com/michaelasper/kir-ai/compare/v1.8.11...v1.8.12) (2026-05-13)

### Bug Fixes

* **native:** seed sampling rng per request ([3db8a7c](https://github.com/michaelasper/kir-ai/commit/3db8a7c78935a6da1412506ab8d448efa8f9066f))

## [1.8.11](https://github.com/michaelasper/kir-ai/compare/v1.8.10...v1.8.11) (2026-05-13)

### Bug Fixes

* **bench:** report focused qwen agentic metrics ([e26540d](https://github.com/michaelasper/kir-ai/commit/e26540ddcc177e0427942a168fb15689d37c31e4))

## [1.8.10](https://github.com/michaelasper/kir-ai/compare/v1.8.9...v1.8.10) (2026-05-13)

### Bug Fixes

* **embeddings:** validate token ids before lookup ([c536d6d](https://github.com/michaelasper/kir-ai/commit/c536d6da3f0fd8b32ef5cf2f04ec30b07d94c69c))

## [1.8.9](https://github.com/michaelasper/kir-ai/compare/v1.8.8...v1.8.9) (2026-05-13)

### Bug Fixes

* **metal:** check buffer byte lengths ([98014ee](https://github.com/michaelasper/kir-ai/commit/98014ee3c80753cb9b0e0e87dd01d8fcc9116f02))

## [1.8.8](https://github.com/michaelasper/kir-ai/compare/v1.8.7...v1.8.8) (2026-05-13)

### Bug Fixes

* **metal:** share command queue across kernels ([a74bf8c](https://github.com/michaelasper/kir-ai/commit/a74bf8c7e9fc493400930bdbcf3119c4f8f32970))

## [1.8.7](https://github.com/michaelasper/kir-ai/compare/v1.8.6...v1.8.7) (2026-05-13)

### Bug Fixes

* **metal:** bucket scratch buffers by size ([fa38323](https://github.com/michaelasper/kir-ai/commit/fa38323f9d5748cce3911328e04e94dbcf364af5))

## [1.8.6](https://github.com/michaelasper/kir-ai/compare/v1.8.5...v1.8.6) (2026-05-13)

### Bug Fixes

* **metal:** cache command buffer timeout ([747cc4b](https://github.com/michaelasper/kir-ai/commit/747cc4bdc07abfc7a71422a76d5bd153437f10e0))

## [1.8.5](https://github.com/michaelasper/kir-ai/compare/v1.8.4...v1.8.5) (2026-05-13)

### Bug Fixes

* **attention:** use flat rows for batched prefill ([9e2f6e2](https://github.com/michaelasper/kir-ai/commit/9e2f6e2084aae29777a3a391870c22c1ce50ef55)), closes [#233](https://github.com/michaelasper/kir-ai/issues/233)

## [1.8.4](https://github.com/michaelasper/kir-ai/compare/v1.8.3...v1.8.4) (2026-05-13)

### Bug Fixes

* **scheduler:** estimate tool sizes without JSON allocation ([e1310fd](https://github.com/michaelasper/kir-ai/commit/e1310fd51d17ef9eddd60e6ceb4c1c47f2d2a269)), closes [#228](https://github.com/michaelasper/kir-ai/issues/228)

## [1.8.3](https://github.com/michaelasper/kir-ai/compare/v1.8.2...v1.8.3) (2026-05-13)

### Bug Fixes

* **native-text:** preserve shifted streaming prefixes ([cc4886d](https://github.com/michaelasper/kir-ai/commit/cc4886d06be900cde6adc6b32b1c92af6c832f6d)), closes [#232](https://github.com/michaelasper/kir-ai/issues/232)

## [1.8.2](https://github.com/michaelasper/kir-ai/compare/v1.8.1...v1.8.2) (2026-05-13)

### Bug Fixes

* **native-text:** precompute stop token ids ([128eece](https://github.com/michaelasper/kir-ai/commit/128eece73cb2455ff4aab426c0ef29d0b493ece5)), closes [#243](https://github.com/michaelasper/kir-ai/issues/243)

## [1.8.1](https://github.com/michaelasper/kir-ai/compare/v1.8.0...v1.8.1) (2026-05-13)

### Bug Fixes

* **native-text:** reduce prefix cache lock contention ([4f1f2ea](https://github.com/michaelasper/kir-ai/commit/4f1f2ea8aef642c3bfa2af62beac1061137990d0)), closes [#133](https://github.com/michaelasper/kir-ai/issues/133)

## [1.8.0](https://github.com/michaelasper/kir-ai/compare/v1.7.1...v1.8.0) (2026-05-13)

### Features

* **streaming:** add structured tool-call fast path ([89b5091](https://github.com/michaelasper/kir-ai/commit/89b50913198200151bf15e4f4ed87385965784d7))

## [1.7.1](https://github.com/michaelasper/kir-ai/compare/v1.7.0...v1.7.1) (2026-05-13)

### Bug Fixes

* **bench:** use server default for mlx sweep lanes ([b8b2d33](https://github.com/michaelasper/kir-ai/commit/b8b2d331ada87bd6a3f695aeca8015012d29a7d3)), closes [#255](https://github.com/michaelasper/kir-ai/issues/255)

## [1.7.0](https://github.com/michaelasper/kir-ai/compare/v1.6.0...v1.7.0) (2026-05-13)

### Features

* **bench:** add qwen mlx cache prefill sweep ([c7c9e7a](https://github.com/michaelasper/kir-ai/commit/c7c9e7ac5f288d843a26d3725bc6492e508f3c80)), closes [#250](https://github.com/michaelasper/kir-ai/issues/250) [#252](https://github.com/michaelasper/kir-ai/issues/252)

## [1.6.0](https://github.com/michaelasper/kir-ai/compare/v1.5.0...v1.6.0) (2026-05-13)

### Features

* **bench:** add canonical Qwen MLX tool benchmark ([5429e83](https://github.com/michaelasper/kir-ai/commit/5429e83ebd0df04297db90f0188b74a3683d52ee))

## [1.5.0](https://github.com/michaelasper/kir-ai/compare/v1.4.37...v1.5.0) (2026-05-12)

### Features

* **metrics:** split mlx sidecar latency ([5b9d4bb](https://github.com/michaelasper/kir-ai/commit/5b9d4bba2daad2cd1fe30bfd041c133c9aa10ca7))

## [1.4.37](https://github.com/michaelasper/kir-ai/compare/v1.4.36...v1.4.37) (2026-05-12)

### Bug Fixes

* **mlx:** use non-streaming upstream for blocking generate ([4d85734](https://github.com/michaelasper/kir-ai/commit/4d85734b354973650879896d8f7743bbbb5ff62a))

## [1.4.36](https://github.com/michaelasper/kir-ai/compare/v1.4.35...v1.4.36) (2026-05-12)

### Bug Fixes

* **runtime:** normalize json object responses ([336043a](https://github.com/michaelasper/kir-ai/commit/336043a101ef84e42c92bc67c2ea35f3bfc72c3b))

## [1.4.35](https://github.com/michaelasper/kir-ai/compare/v1.4.34...v1.4.35) (2026-05-12)

### Bug Fixes

* **engine:** accept legacy deterministic backend flag ([52a060a](https://github.com/michaelasper/kir-ai/commit/52a060a39f7502b7bc1e22b660acbb4157eb3d94))

## [1.4.34](https://github.com/michaelasper/kir-ai/compare/v1.4.33...v1.4.34) (2026-05-12)

### Bug Fixes

* **metal:** batch full attention cache mix ([ee45c5b](https://github.com/michaelasper/kir-ai/commit/ee45c5b7b8abf8a38153f44fd2bd38dff7ec19d0)), closes [#225](https://github.com/michaelasper/kir-ai/issues/225)

## [1.4.33](https://github.com/michaelasper/kir-ai/compare/v1.4.32...v1.4.33) (2026-05-12)

### Bug Fixes

* **protocol:** parameterize protocol test backend family ([4fd7f39](https://github.com/michaelasper/kir-ai/commit/4fd7f391add352f0c33c41d7122b437a22cfef27)), closes [#223](https://github.com/michaelasper/kir-ai/issues/223)

## [1.4.32](https://github.com/michaelasper/kir-ai/compare/v1.4.31...v1.4.32) (2026-05-12)

### Bug Fixes

* **native-text:** decode streaming tokens incrementally ([e7c3586](https://github.com/michaelasper/kir-ai/commit/e7c3586135d5efdb221066598119b1ecc542970b)), closes [#221](https://github.com/michaelasper/kir-ai/issues/221)

## [1.4.31](https://github.com/michaelasper/kir-ai/compare/v1.4.30...v1.4.31) (2026-05-12)

### Bug Fixes

* **native-text:** reuse worker runtime and shallow driver clones ([2f9ba01](https://github.com/michaelasper/kir-ai/commit/2f9ba01f1ebaa11df439dca59b20226f6061f354)), closes [#230](https://github.com/michaelasper/kir-ai/issues/230)

## [1.4.30](https://github.com/michaelasper/kir-ai/compare/v1.4.29...v1.4.30) (2026-05-12)

### Bug Fixes

* **metal:** report dropped command buffer status ([24a8261](https://github.com/michaelasper/kir-ai/commit/24a8261608183b799d708af89303bd199c5dcbd4)), closes [#234](https://github.com/michaelasper/kir-ai/issues/234)

## [1.4.29](https://github.com/michaelasper/kir-ai/compare/v1.4.28...v1.4.29) (2026-05-12)

### Bug Fixes

* **engine:** enforce stream stall progress deadline ([82b85d1](https://github.com/michaelasper/kir-ai/commit/82b85d13f2a6f3600f64f618e82420fc6d2a3cf1)), closes [#229](https://github.com/michaelasper/kir-ai/issues/229)

## [1.4.28](https://github.com/michaelasper/kir-ai/compare/v1.4.27...v1.4.28) (2026-05-12)

### Bug Fixes

* **parser:** tolerate truncated qwen reasoning ([7f23768](https://github.com/michaelasper/kir-ai/commit/7f237685aa5eeaa4097aad51bfcb811dc1ad75a8)), closes [#227](https://github.com/michaelasper/kir-ai/issues/227)

## [1.4.27](https://github.com/michaelasper/kir-ai/compare/v1.4.26...v1.4.27) (2026-05-12)

### Bug Fixes

* **runtime:** fill missing omp intent argument ([e6aa2c8](https://github.com/michaelasper/kir-ai/commit/e6aa2c84e5474439c04699d5bb538f272f79e11e)), closes [#215](https://github.com/michaelasper/kir-ai/issues/215)

## [1.4.26](https://github.com/michaelasper/kir-ai/compare/v1.4.25...v1.4.26) (2026-05-12)

### Bug Fixes

* **runtime:** stream safe unmarked llama text ([f611f54](https://github.com/michaelasper/kir-ai/commit/f611f54e4ec527ba05c5875ce6a4b9514155c4b9)), closes [#188](https://github.com/michaelasper/kir-ai/issues/188)

## [1.4.25](https://github.com/michaelasper/kir-ai/compare/v1.4.24...v1.4.25) (2026-05-12)

### Bug Fixes

* **parser:** narrow auto xlam detection ([5ac7e49](https://github.com/michaelasper/kir-ai/commit/5ac7e49cbd7c9513063826d8ab3146cb0be1ddfb)), closes [#204](https://github.com/michaelasper/kir-ai/issues/204)

## [1.4.24](https://github.com/michaelasper/kir-ai/compare/v1.4.23...v1.4.24) (2026-05-12)

### Performance Improvements

* **backend:** make cpu in-place ops allocation-free ([3f1cf1e](https://github.com/michaelasper/kir-ai/commit/3f1cf1e2b1c97a979fb04865479968789d5edef1)), closes [#202](https://github.com/michaelasper/kir-ai/issues/202)

## [1.4.23](https://github.com/michaelasper/kir-ai/compare/v1.4.22...v1.4.23) (2026-05-12)

### Performance Improvements

* **sampler:** reuse top-p sampling scratch ([e4ddda6](https://github.com/michaelasper/kir-ai/commit/e4ddda6cb236cec21ddbd86789396aa599516e2a)), closes [#196](https://github.com/michaelasper/kir-ai/issues/196)

## [1.4.22](https://github.com/michaelasper/kir-ai/compare/v1.4.21...v1.4.22) (2026-05-12)

### Performance Improvements

* **metal:** reuse matvec scratch buffers ([c1cb493](https://github.com/michaelasper/kir-ai/commit/c1cb4930ad4a1af7a0d2f4090a818084085a3245)), closes [#141](https://github.com/michaelasper/kir-ai/issues/141)

## [1.4.21](https://github.com/michaelasper/kir-ai/compare/v1.4.20...v1.4.21) (2026-05-12)

### Performance Improvements

* **metal:** parallelize softmax reduction ([c6084f0](https://github.com/michaelasper/kir-ai/commit/c6084f0836722a2cbc394567278fda9e7ae83b9c)), closes [#172](https://github.com/michaelasper/kir-ai/issues/172)

## [1.4.20](https://github.com/michaelasper/kir-ai/compare/v1.4.19...v1.4.20) (2026-05-12)

### Performance Improvements

* **metal:** reduce rms norm in one threadgroup ([18aebea](https://github.com/michaelasper/kir-ai/commit/18aebeaf756dbe663b1defea3a0d993fdace09d3)), closes [#174](https://github.com/michaelasper/kir-ai/issues/174)

## [1.4.19](https://github.com/michaelasper/kir-ai/compare/v1.4.18...v1.4.19) (2026-05-12)

### Bug Fixes

* **qwen:** avoid transient rms norm weight allocation ([c26df7a](https://github.com/michaelasper/kir-ai/commit/c26df7a32d2b7ec6f86cadd1915bf12179b26892)), closes [#155](https://github.com/michaelasper/kir-ai/issues/155)

## [1.4.18](https://github.com/michaelasper/kir-ai/compare/v1.4.17...v1.4.18) (2026-05-12)

### Bug Fixes

* **metal:** time out command buffer waits ([253ed42](https://github.com/michaelasper/kir-ai/commit/253ed427958d39d29eec9ffed4bdda0f00291e87)), closes [#192](https://github.com/michaelasper/kir-ai/issues/192)

## [1.4.17](https://github.com/michaelasper/kir-ai/compare/v1.4.16...v1.4.17) (2026-05-12)

### Bug Fixes

* **hub:** serialize model pull promotion ([2ed7cd9](https://github.com/michaelasper/kir-ai/commit/2ed7cd93b44a9d869c952d0fc9e01f4989d448f9)), closes [#181](https://github.com/michaelasper/kir-ai/issues/181)

## [1.4.16](https://github.com/michaelasper/kir-ai/compare/v1.4.15...v1.4.16) (2026-05-12)

### Bug Fixes

* **runtime:** scope unmarked JSON truncation tokens ([191f50e](https://github.com/michaelasper/kir-ai/commit/191f50e165c2a286607ae2bda1f281365770cc2b)), closes [#178](https://github.com/michaelasper/kir-ai/issues/178)

## [1.4.15](https://github.com/michaelasper/kir-ai/compare/v1.4.14...v1.4.15) (2026-05-12)

### Bug Fixes

* **hub:** require weight sha256 verification ([1ed93a0](https://github.com/michaelasper/kir-ai/commit/1ed93a007a1cc08485f1f4b6775573cd29092c68)), closes [#177](https://github.com/michaelasper/kir-ai/issues/177)

## [1.4.14](https://github.com/michaelasper/kir-ai/compare/v1.4.13...v1.4.14) (2026-05-12)

### Performance Improvements

* **models:** avoid reparsing inferred native configs ([7251374](https://github.com/michaelasper/kir-ai/commit/72513743e77f9e799b3aaf1c63dafc2705988f19)), closes [#186](https://github.com/michaelasper/kir-ai/issues/186)

## [1.4.13](https://github.com/michaelasper/kir-ai/compare/v1.4.12...v1.4.13) (2026-05-12)

### Bug Fixes

* **models:** remove qwen-specific promotion stage ([58cbff4](https://github.com/michaelasper/kir-ai/commit/58cbff4651c16915db2c5d1f5a93345881f2b1b6)), closes [#182](https://github.com/michaelasper/kir-ai/issues/182)

## [1.4.12](https://github.com/michaelasper/kir-ai/compare/v1.4.11...v1.4.12) (2026-05-12)

### Bug Fixes

* **models:** align backend execution capabilities ([a912139](https://github.com/michaelasper/kir-ai/commit/a91213996f7c5d3d8408b55dde35296161c271dd)), closes [#184](https://github.com/michaelasper/kir-ai/issues/184)

## [1.4.11](https://github.com/michaelasper/kir-ai/compare/v1.4.10...v1.4.11) (2026-05-12)

### Bug Fixes

* **engine:** fail closed on poisoned locks ([e38683e](https://github.com/michaelasper/kir-ai/commit/e38683ec78d19ff4f9455387d0ce5812d5e54c72)), closes [#149](https://github.com/michaelasper/kir-ai/issues/149)

## [1.4.10](https://github.com/michaelasper/kir-ai/compare/v1.4.9...v1.4.10) (2026-05-12)

### Bug Fixes

* **engine:** require protocol backend acknowledgement ([4391650](https://github.com/michaelasper/kir-ai/commit/43916507c7d10d5cfcc7ee693748dc619f7afe63)), closes [#152](https://github.com/michaelasper/kir-ai/issues/152)

## [1.4.9](https://github.com/michaelasper/kir-ai/compare/v1.4.8...v1.4.9) (2026-05-12)

### Bug Fixes

* **engine:** return router config errors ([7adc7ec](https://github.com/michaelasper/kir-ai/commit/7adc7ecbf0140ace4eb84f229d8569e2c3ecd808)), closes [#148](https://github.com/michaelasper/kir-ai/issues/148)

## [1.4.8](https://github.com/michaelasper/kir-ai/compare/v1.4.7...v1.4.8) (2026-05-12)

### Bug Fixes

* **tools:** align tool argument keys ([468c95f](https://github.com/michaelasper/kir-ai/commit/468c95f8e2798d6875ebcfa32d3dd30b8c2554d3)), closes [#180](https://github.com/michaelasper/kir-ai/issues/180)

## [1.4.7](https://github.com/michaelasper/kir-ai/compare/v1.4.6...v1.4.7) (2026-05-12)

### Bug Fixes

* **api:** bound inference request sizes ([7d5a055](https://github.com/michaelasper/kir-ai/commit/7d5a055123f64f97fdbf0bb776f20560cf44f49b)), closes [#153](https://github.com/michaelasper/kir-ai/issues/153)

## [1.4.6](https://github.com/michaelasper/kir-ai/compare/v1.4.5...v1.4.6) (2026-05-12)

### Bug Fixes

* **mlx:** bound sidecar request timeouts ([f1630c8](https://github.com/michaelasper/kir-ai/commit/f1630c8c0ca64160f0c69e6e2637dd91944be86c)), closes [#134](https://github.com/michaelasper/kir-ai/issues/134)

## [1.4.5](https://github.com/michaelasper/kir-ai/compare/v1.4.4...v1.4.5) (2026-05-12)

### Bug Fixes

* **metal:** synchronize shared buffer access ([edeae67](https://github.com/michaelasper/kir-ai/commit/edeae67fb5b269cc21e4bd78065f8de23815db08)), closes [#175](https://github.com/michaelasper/kir-ai/issues/175)

## [1.4.4](https://github.com/michaelasper/kir-ai/compare/v1.4.3...v1.4.4) (2026-05-12)

### Bug Fixes

* **engine:** run native generation off async workers ([b67e339](https://github.com/michaelasper/kir-ai/commit/b67e3397220c7a0bf046cc2374aa36ea8915fedb)), closes [#144](https://github.com/michaelasper/kir-ai/issues/144)

## [1.4.3](https://github.com/michaelasper/kir-ai/compare/v1.4.2...v1.4.3) (2026-05-12)

### Bug Fixes

* **engine:** require explicit admin auth opt-in ([77c166e](https://github.com/michaelasper/kir-ai/commit/77c166e116516b370d27e457982013311eeeddc6)), closes [#145](https://github.com/michaelasper/kir-ai/issues/145) [#146](https://github.com/michaelasper/kir-ai/issues/146)

## [1.4.2](https://github.com/michaelasper/kir-ai/compare/v1.4.1...v1.4.2) (2026-05-12)

### Bug Fixes

* **mlx:** preserve lossless tool chat history ([fbd8305](https://github.com/michaelasper/kir-ai/commit/fbd8305c7903784b888df9a1fbcb3ca5d3489d00))

## [1.4.1](https://github.com/michaelasper/kir-ai/compare/v1.4.0...v1.4.1) (2026-05-11)

### Bug Fixes

* filter tool messages instead of dropping entire chat context ([357860d](https://github.com/michaelasper/kir-ai/commit/357860de289dd1d642e0ae9e8b05167986a794de))

## [1.4.0](https://github.com/michaelasper/kir-ai/compare/v1.3.2...v1.4.0) (2026-05-11)

### Features

* **mlx:** add --mlx-{connect,request,read}-timeout CLI flags ([f4dc107](https://github.com/michaelasper/kir-ai/commit/f4dc1075e95621922a8ea92bf8e4da8a73bc7b66))
* **mlx:** add MlxTimeouts type and build_http_client helper ([65f7a50](https://github.com/michaelasper/kir-ai/commit/65f7a50c0b0fafaa744c0c95631734192ae821db))
* **mlx:** add per-chunk read timeout to stream_completion ([b07c42b](https://github.com/michaelasper/kir-ai/commit/b07c42b961898a198c6b2a4d46a2bbd12ee183a5))
* **mlx:** add Stall failure metric kind ([f8c2b9e](https://github.com/michaelasper/kir-ai/commit/f8c2b9eb148f80cb5fe89d5c370c1f4d77eb523a))
* **mlx:** wire MlxTimeouts into MlxBackendOptions and client construction ([46f286f](https://github.com/michaelasper/kir-ai/commit/46f286f8a97434ee004ae3b2af61542fca8facf4))

### Bug Fixes

* **mlx:** remove overall request timeout, use sentinel prefix for stall classification ([f616804](https://github.com/michaelasper/kir-ai/commit/f616804392add9430827a53f83c994862ed4c8be))

## [1.3.2](https://github.com/michaelasper/kir-ai/compare/v1.3.1...v1.3.2) (2026-05-11)

### Bug Fixes

* **ci:** add --all-features to engine_model_cli_contracts gate ([f7cd043](https://github.com/michaelasper/kir-ai/commit/f7cd043f1c2f8ebfd80870339b628a353364121c))
* **ci:** add --all-features to north-star gate test commands ([6b6968d](https://github.com/michaelasper/kir-ai/commit/6b6968def601f80e7502583a5f4baa5b7e1db426))

## [1.3.1](https://github.com/michaelasper/kir-ai/compare/v1.3.0...v1.3.1) (2026-05-11)

### Bug Fixes

* **runtime:** eliminate duplicate content emission in streaming_chat_stream ([d907aa0](https://github.com/michaelasper/kir-ai/commit/d907aa0f1b6327053014261544c39719f8b99c91)), closes [#150](https://github.com/michaelasper/kir-ai/issues/150)

## [1.3.0](https://github.com/michaelasper/kir-ai/compare/v1.2.5...v1.3.0) (2026-05-11)

### Features

* **engine:** add shared async file I/O helpers for tokio-safe reads ([47c286c](https://github.com/michaelasper/kir-ai/commit/47c286cfd23090a91f8c7f095ffa451b3ec409f2))
* **engine:** convert inspect-qwen-input CLI path to tokio::fs ([5954f42](https://github.com/michaelasper/kir-ai/commit/5954f42d6f11a3c0ef884113f36733d16bbcf5f2)), closes [#136](https://github.com/michaelasper/kir-ai/issues/136)

### Performance Improvements

* **backend:** add f32 tensor cache to SafeTensorShardStore ([22aadd9](https://github.com/michaelasper/kir-ai/commit/22aadd94f1073c4310cc2d374b2edf24e9b7c93b)), closes [#147](https://github.com/michaelasper/kir-ai/issues/147)

## [1.2.5](https://github.com/michaelasper/kir-ai/compare/v1.2.4...v1.2.5) (2026-05-11)

### Bug Fixes

* **qwen:** correct query/key handling in linear and full attention decode steps ([a5a1d27](https://github.com/michaelasper/kir-ai/commit/a5a1d273abaec7a584f1f58c5f501fdab3396ce9)), closes [#173](https://github.com/michaelasper/kir-ai/issues/173) [#176](https://github.com/michaelasper/kir-ai/issues/176)

## [1.2.4](https://github.com/michaelasper/kir-ai/compare/v1.2.3...v1.2.4) (2026-05-11)

### Bug Fixes

* **sampling:** validate temperature/top_p ranges in SamplingConfig::from_openai_controls ([20b0dbc](https://github.com/michaelasper/kir-ai/commit/20b0dbc3dd54545b1cde6635783293043108a968)), closes [#160](https://github.com/michaelasper/kir-ai/issues/160)

## [1.2.3](https://github.com/michaelasper/kir-ai/compare/v1.2.2...v1.2.3) (2026-05-11)

### Bug Fixes

* **security:** gate ProtocolTestBackend behind non-default test-utils feature ([96a938b](https://github.com/michaelasper/kir-ai/commit/96a938bcae5e5c12a0d3f9558a35c216098bd6d7)), closes [#139](https://github.com/michaelasper/kir-ai/issues/139)

## [1.2.2](https://github.com/michaelasper/kir-ai/compare/v1.2.1...v1.2.2) (2026-05-11)

### Bug Fixes

* **runtime:** use family-aware tool markers in no-progress classifier ([204f981](https://github.com/michaelasper/kir-ai/commit/204f9814bb7a3a76ebd3587b20c073dabdfc0687)), closes [#154](https://github.com/michaelasper/kir-ai/issues/154)

## [1.2.1](https://github.com/michaelasper/kir-ai/compare/v1.2.0...v1.2.1) (2026-05-11)

### Bug Fixes

* **safety:** replace production panics with Result-based error handling ([fd9b616](https://github.com/michaelasper/kir-ai/commit/fd9b6169ea58f38779fbf1aad12559094d50f73e)), closes [#137](https://github.com/michaelasper/kir-ai/issues/137)

## [1.2.0](https://github.com/michaelasper/kir-ai/compare/v1.1.0...v1.2.0) (2026-05-11)

### Features

* **api:** implement review suggestions for structured errors and CI schema validation ([752bb04](https://github.com/michaelasper/kir-ai/commit/752bb045bc9a78bdccabbd57ea25cb0e9e505d39))

## [1.1.0](https://github.com/michaelasper/kir-ai/compare/v1.0.1...v1.1.0) (2026-05-11)

### Features

* **api:** document and structure admin API responses with schemars ([47d12c6](https://github.com/michaelasper/kir-ai/commit/47d12c63735bc99ed57cdb5282ad6be01dcccfe6))

## [1.0.1](https://github.com/michaelasper/kir-ai/compare/v1.0.0...v1.0.1) (2026-05-10)

### Bug Fixes

* **ci:** fix formatting and clippy lints across the workspace ([b5ce81e](https://github.com/michaelasper/kir-ai/commit/b5ce81e7ab7a3236205a4fe09bd6cf5242848481))

## 1.0.0 (2026-05-10)

### Features

* add admin model plan and pull endpoints ([f9fe943](https://github.com/michaelasper/kir-ai/commit/f9fe9435e65d4bc95ab18ccdf85bbfd9220b9147))
* add admin request cancellation ([c48be82](https://github.com/michaelasper/kir-ai/commit/c48be82c1c49bcf7a126c16264ad8d706e3f836f))
* add batched metal bf16 matvec ([a316fbb](https://github.com/michaelasper/kir-ai/commit/a316fbbf8be54d56e84f47dac2fe36ce628c5f87))
* add full attention cache decode step ([b1b8f82](https://github.com/michaelasper/kir-ai/commit/b1b8f8231d67026c4c59869b91449144fba967f4))
* add layer kv cache storage ([572aaab](https://github.com/michaelasper/kir-ai/commit/572aaabfd5e1ca391eb8b3fb632c9bd89261b506))
* add linear attention cache decode step ([c2ec88f](https://github.com/michaelasper/kir-ai/commit/c2ec88f16afc6469493023ce009a866b650e89eb))
* add linear attention decode cache ([1cd0161](https://github.com/michaelasper/kir-ai/commit/1cd0161d00e4e516f52ca8e194aecb55de2e6f8a))
* add llama mlx chat family ([a37abe9](https://github.com/michaelasper/kir-ai/commit/a37abe9f05faadb1b5b34fae4b575572483e1572))
* add metal bf16 matvec kernel ([86f64ff](https://github.com/michaelasper/kir-ai/commit/86f64ff312faa4d8b17734799ab4a0c50a8eb1db))
* add metal f32 logits kernels ([a6f92f3](https://github.com/michaelasper/kir-ai/commit/a6f92f3421dd8d1e07f7960af56c93886ff4b6ad))
* add metal f32 matvec kernel ([2e7c802](https://github.com/michaelasper/kir-ai/commit/2e7c802d9e9289900916f0cc92ed37213825099b))
* add model prune dry run ([dbeffc8](https://github.com/michaelasper/kir-ai/commit/dbeffc8825b3baa6c3b053c07c95311e8a8bd7a3))
* add native hugging face model planning ([3630acd](https://github.com/michaelasper/kir-ai/commit/3630acda3e020fe7ed293078fd175940b0dd1fcc))
* add native model store pull ([c5f147c](https://github.com/michaelasper/kir-ai/commit/c5f147ca75f2aeecd8eda818f5230d83b488da03))
* add qwen full attention sequence math ([2e0a54a](https://github.com/michaelasper/kir-ai/commit/2e0a54ae4a1a49fecf2f33b1742939c6e1920f15))
* add qwen linear sequence recurrence ([8593a74](https://github.com/michaelasper/kir-ai/commit/8593a741208b1304e84dad2cdde8b374c66e4246))
* add qwen rmsnorm metal kernel ([c0d8a74](https://github.com/michaelasper/kir-ai/commit/c0d8a74d2cb5d46d3a0a9a1948829b34dde440a5))
* add qwen shard prefill layers ([d7c10eb](https://github.com/michaelasper/kir-ai/commit/d7c10ebe55a50de1e553e0d2d6adde6b788a724f))
* add rust openai runtime skeleton ([9dae41e](https://github.com/michaelasper/kir-ai/commit/9dae41ec517ff2c15b0ec0b9a4342aa4d9a1d557))
* add safetensors and metal smoke kernels ([52c78bc](https://github.com/michaelasper/kir-ai/commit/52c78bceeb31ab1faefe8f7fd04c2d267aa8b12d))
* add text completions endpoint ([7b96723](https://github.com/michaelasper/kir-ai/commit/7b9672310f3610f07a57f108b3e849eb7bda1006))
* add top-p sampler primitive ([f80974c](https://github.com/michaelasper/kir-ai/commit/f80974c7527c1ca542d6b61211f16369cd452638))
* allocate qwen layer caches ([9934f3e](https://github.com/michaelasper/kir-ai/commit/9934f3e95c3ecb9194eecc937723f53e6cbff3e9))
* apply chat stop sequences ([2c8d4b5](https://github.com/michaelasper/kir-ai/commit/2c8d4b5770c632f3ae6fda9be28ad0af897d5957))
* cache full attention prefill ([ac91d0b](https://github.com/michaelasper/kir-ai/commit/ac91d0bcb3988f593d1362db5fcbbcf403c7ac24))
* cache linear attention prefill ([a77c129](https://github.com/michaelasper/kir-ai/commit/a77c129500e6d3b90b58cb0ea790403d1e80d0e3))
* cache qwen full attention layer prefill ([6535150](https://github.com/michaelasper/kir-ai/commit/6535150b9af203855a9eadd44b07cf17155746ab))
* cache qwen linear attention layer prefill ([30a9583](https://github.com/michaelasper/kir-ai/commit/30a95834d51452e10b782fbe71f46fca2178da79))
* compute qwen layer0 linear attention ([38cc5fd](https://github.com/michaelasper/kir-ai/commit/38cc5fdd5b63af61c3c96dd034dfb5a35b005bc2))
* decode qwen full attention layer with cache ([79352bc](https://github.com/michaelasper/kir-ai/commit/79352bc2696101aca321bb54d2fd12334a9ce80f))
* decode qwen full pass logits ([d4cb27d](https://github.com/michaelasper/kir-ai/commit/d4cb27dee4530093ea44514ede481135f9a75106))
* decode qwen linear attention layer with cache ([c1dbb12](https://github.com/michaelasper/kir-ai/commit/c1dbb12cc109982527dd4ba94748427727af0430))
* decode qwen token with layer caches ([1d111cb](https://github.com/michaelasper/kir-ai/commit/1d111cb9c3f29ead79927a054c961633dddb127e))
* eager materialize native qwen shards ([311325f](https://github.com/michaelasper/kir-ai/commit/311325fa1fa7334c4adc038973b9af531fcffb6e))
* enforce json object response mode ([24fca9c](https://github.com/michaelasper/kir-ai/commit/24fca9cf077d090be937e49ea9d32379b636d5ce))
* execute qwen layer0 moe slices ([f0cfbfa](https://github.com/michaelasper/kir-ai/commit/f0cfbfa16a84d9943874e4ad8699557a42655f74))
* expand gemma native inference and tooling ([b43300f](https://github.com/michaelasper/kir-ai/commit/b43300f5fade29313711955392cde2704cac2486))
* expose aggregate inference metrics ([17e3fd8](https://github.com/michaelasper/kir-ai/commit/17e3fd830c80caddcef60fdbac4a54733ba5fda6))
* expose backend model metadata ([358196e](https://github.com/michaelasper/kir-ai/commit/358196e0f7edbf881a5124ae9e5b510307b05e40))
* expose mlx backend metrics ([80718ba](https://github.com/michaelasper/kir-ai/commit/80718ba1b63ba6945328f393ffd5e271ab6048e5))
* expose one-step kirai install flow ([055da8a](https://github.com/michaelasper/kir-ai/commit/055da8ad8b4cea161f8f0788c2b83f6c1c9a3a52))
* expose raw bf16 tensor ranges ([60d5328](https://github.com/michaelasper/kir-ai/commit/60d5328a005143143591b8b8df5dcff443648548))
* expose read-only admin model status ([1c8faef](https://github.com/michaelasper/kir-ai/commit/1c8faefd57c9dcff07c3e224f18d5bc25d1332f4))
* implement asynchronous metal execution pipeline ([7f99e11](https://github.com/michaelasper/kir-ai/commit/7f99e11131d7ce8ea5dd7bfca1d182cb001d0443))
* include failure phase and retryability ([4cf4cf6](https://github.com/michaelasper/kir-ai/commit/4cf4cf60a4f2cec8f0148ae7e5f9da9f69b70448))
* include usage in requested streams ([fdafefd](https://github.com/michaelasper/kir-ai/commit/fdafefd930eaf23a31d237c96ed69fdd8860c37e))
* inspect and verify local model snapshots ([ca7c097](https://github.com/michaelasper/kir-ai/commit/ca7c0975203b2cb026e9f40b857065e33ec1fbf1))
* inspect safetensors headers natively ([3d46d5a](https://github.com/michaelasper/kir-ai/commit/3d46d5a71b2fc4e2243755d9fbf291f9edfa2c89))
* list local model snapshots ([bf5ac3e](https://github.com/michaelasper/kir-ai/commit/bf5ac3eabd902d9b15e02c5e192d2934b5b691c3))
* load qwen36 tokenizer artifact ([ccab198](https://github.com/michaelasper/kir-ai/commit/ccab19824bdd1380324cb04fe9c35bbaf1caaaa5))
* materialize all safetensor shards ([bd20f2f](https://github.com/michaelasper/kir-ai/commit/bd20f2feec980e7826dfbb41a0aaa2b1971288ce))
* mmap safetensors shard reads ([c8f46e8](https://github.com/michaelasper/kir-ai/commit/c8f46e89c33db5e4a436992ae9f0576b86c66d90))
* parse qwen36 model metadata ([1f6488e](https://github.com/michaelasper/kir-ai/commit/1f6488e24fe60539da7ccddef4f49081061641e5))
* probe qwen embeddings natively ([3f74400](https://github.com/michaelasper/kir-ai/commit/3f74400acf8bbb845507b9bad1937c32558b6281))
* project qwen layer0 linear attention inputs ([681d02c](https://github.com/michaelasper/kir-ai/commit/681d02cce1865938a15b9995ab67f1ef697771b0))
* read bf16 tensor ranges ([80b4324](https://github.com/michaelasper/kir-ai/commit/80b432484e455020fac6d173fd28c288b2200b74))
* report active cancellation metrics ([838f218](https://github.com/michaelasper/kir-ai/commit/838f218617c7fdab45e27e872238a15809495d6f))
* report artifact verification metrics ([392da7b](https://github.com/michaelasper/kir-ai/commit/392da7beb3638a734ec1a8cec284521c124fb584))
* report model pull metrics ([4201417](https://github.com/michaelasper/kir-ai/commit/4201417dfe9510181c05c023a1197b1e77b98d29))
* report model store usage metrics ([48a3aa5](https://github.com/michaelasper/kir-ai/commit/48a3aa5c538b883bbf323b6c324fe790285dce3b))
* report no progress metrics ([d72c499](https://github.com/michaelasper/kir-ai/commit/d72c4991c6f86051fac0eec67efb09dbc72d9585))
* report process memory metrics ([a80cd60](https://github.com/michaelasper/kir-ai/commit/a80cd601a80776e3fc88cb0869fce7e6a2b3b282))
* report request latency metrics ([36acd23](https://github.com/michaelasper/kir-ai/commit/36acd23e41ee02b972f0bdac1c8dde302847b9ed))
* report scheduler phase metrics ([b08bdbb](https://github.com/michaelasper/kir-ai/commit/b08bdbb4f9a55b41b5359647599ad7c7b3e997a0))
* report streaming ttft metrics ([4f8cddf](https://github.com/michaelasper/kir-ai/commit/4f8cddfaa767399c8a1190e78a6fb165de3372f4))
* resolve indexed safetensor shards ([3784a5a](https://github.com/michaelasper/kir-ai/commit/3784a5a8411234d809757eccb814275ae209c576))
* resume and verify model downloads ([199bf83](https://github.com/michaelasper/kir-ai/commit/199bf838886b3751d15811f029f4adf8dd9a5a23))
* return stable engine error codes ([6907375](https://github.com/michaelasper/kir-ai/commit/690737599876fddaadddbd7744a41c310e97fe25))
* reuse qwen caches for native decode ([7449bb6](https://github.com/michaelasper/kir-ai/commit/7449bb6616a9bf806269cfeccaed7e916c657d84))
* reuse verified model snapshots ([c65c892](https://github.com/michaelasper/kir-ai/commit/c65c892413daa5369335697e04b29171a699ee11))
* route qwen final norm through executor ([4a5cfbd](https://github.com/michaelasper/kir-ai/commit/4a5cfbd88fb69695ea32d1f5324ae4bc01080529))
* route qwen layer0 moe experts ([e4ea9b6](https://github.com/michaelasper/kir-ai/commit/e4ea9b6023097227f9463b0600691821215a10ee))
* run qwen full attention layers ([289c1de](https://github.com/michaelasper/kir-ai/commit/289c1de6e59dcdbb858b0fa399d1fa4e7d9e6c73))
* run qwen linear decoder layers ([1d52515](https://github.com/michaelasper/kir-ai/commit/1d52515ac78b6115f89439d7c1d6845613c66575))
* run qwen prefill with layer caches ([5cfb4b0](https://github.com/michaelasper/kir-ai/commit/5cfb4b0b6db2f560ec9ebbed6b56feee2c00cef1))
* sample native qwen from full logits ([2cfbf2d](https://github.com/michaelasper/kir-ai/commit/2cfbf2d61950bc6e6d7d658f6b6f85b836616ed6))
* serve native qwen token path ([26ed009](https://github.com/michaelasper/kir-ai/commit/26ed00990ca700936742f9631aab32abb585a80e))
* stream bf16 matvec top logits ([8c8cd47](https://github.com/michaelasper/kir-ai/commit/8c8cd477d61b436608d8104c77419a8f7496ff3d))
* stream chat completions as SSE ([a3120e5](https://github.com/michaelasper/kir-ai/commit/a3120e5505f2677650bb486584385da29d147cc5))
* stream qwen lm head top logits ([39e85ae](https://github.com/michaelasper/kir-ai/commit/39e85ae6584698b8adde8f6473a27f25776df976))
* stream text completions ([4ef489e](https://github.com/michaelasper/kir-ai/commit/4ef489efd1459eab613d68858500f838034cea61))
* stream tool call deltas ([986f580](https://github.com/michaelasper/kir-ai/commit/986f5804c4b22a4a03100fd5e7c3f44e825a57de))
* support chat max completion tokens ([1b76a55](https://github.com/michaelasper/kir-ai/commit/1b76a55a8d18d9ba979232352e9a0c8ac7a1303a))
* use Metal matvecs in native Qwen serving ([5fb7d9e](https://github.com/michaelasper/kir-ai/commit/5fb7d9eed5e0d0bc089def986c3692e13c9ecf3f)), closes [#41](https://github.com/michaelasper/kir-ai/issues/41)
* use qwen bounded prefill backend ([6bb686d](https://github.com/michaelasper/kir-ai/commit/6bb686d4043f3c70b361f1138f9c9c3343e3d428))
* validate qwen36 weight index ([becf073](https://github.com/michaelasper/kir-ai/commit/becf0731d2bc3add091ff7eccf1990660d999d63))
* verify served model snapshots ([90e4988](https://github.com/michaelasper/kir-ai/commit/90e4988fedd938656210480bdace5575eff6b22c))
* wire native sampling controls ([e0bc485](https://github.com/michaelasper/kir-ai/commit/e0bc485e29bb1b3ef19a06ce5cd6a038ec9c4b37))

### Bug Fixes

* accept OpenAI text content parts ([07073a1](https://github.com/michaelasper/kir-ai/commit/07073a1f57d4af1969ef0484663ab5a5816e43dc))
* adapt deterministic chat protocol responses ([bfb20f0](https://github.com/michaelasper/kir-ai/commit/bfb20f0fa9af7bdafab93a59d3a998a220dfc5d9))
* add deterministic json object protocol ([66f1dd2](https://github.com/michaelasper/kir-ai/commit/66f1dd250798d8c9fc43229c0d59ab859e7f8ae6))
* add deterministic required tool calls ([12d8cdf](https://github.com/michaelasper/kir-ai/commit/12d8cdfd96dd87707aa2cc842401de18a73d15ff))
* add model concurrency backpressure ([913e25b](https://github.com/michaelasper/kir-ai/commit/913e25b870790bfdf83bcc43989c938ccbd90835))
* address github issues 45 through 47 ([773dac0](https://github.com/michaelasper/kir-ai/commit/773dac0406a95c03a68097b21d5d7a51e67bf0ad))
* address open github issues ([53e7fcd](https://github.com/michaelasper/kir-ai/commit/53e7fcd0af8c9e8d86cb3cc367b3d79b2874914a))
* align sampling controls with OpenAI defaults (close [#170](https://github.com/michaelasper/kir-ai/issues/170)) ([f898f54](https://github.com/michaelasper/kir-ai/commit/f898f54e3ee766e81071bcc616441902be012815))
* allow native qwen snapshots without manifest ([d6bdd00](https://github.com/michaelasper/kir-ai/commit/d6bdd00bdb0b6adfb8926469bb34bfc265845efd))
* apply chat stops before tool parsing ([562b3dd](https://github.com/michaelasper/kir-ai/commit/562b3ddfd66ad8684bccf600e7b720d0451123a3))
* apply qwen centered rmsnorm ([8f9aaa3](https://github.com/michaelasper/kir-ai/commit/8f9aaa33db449b5e0af4bffd580f18b47b421dd2))
* bound hub request and download stalls ([3885568](https://github.com/michaelasper/kir-ai/commit/38855685b223281e1d94cbd432ac53df9232f861))
* cancel backend streams on sse drop ([4e7e4a7](https://github.com/michaelasper/kir-ai/commit/4e7e4a71d4494e77da07c2c5e65d6c5c27b1a6b7))
* cancel non streaming backend generation ([5af89bc](https://github.com/michaelasper/kir-ai/commit/5af89bca764da8eb454a0c387beebb33637ef55a))
* check cancellation during native qwen prefill ([ff60424](https://github.com/michaelasper/kir-ai/commit/ff604245f861a09614693395cfb179d5f0ffb7f5))
* **ci:** add missing conventionalcommits dependency to release workflow ([8fc5eb8](https://github.com/michaelasper/kir-ai/commit/8fc5eb821a4d6abdd3910294143d7234a8ff8785))
* **ci:** align release and installation process with kt ([ddb4d15](https://github.com/michaelasper/kir-ai/commit/ddb4d1538482ac7d8dd7f2ac3f0c146e4bfed996))
* **ci:** remove linux builds and tests as software is apple silicon only ([baccf5b](https://github.com/michaelasper/kir-ai/commit/baccf5ba0cbbf6e6f7f3402d8c5ded69deb14b47))
* classify stream disconnect lifecycle ([eac29c7](https://github.com/michaelasper/kir-ai/commit/eac29c71db3b24475f18b5ec0ea8139efaa7cdd3))
* construct qwen parser directly ([87d560d](https://github.com/michaelasper/kir-ai/commit/87d560d1acab98e19a4331e4ba0c9ea348ac98c4))
* disable qwen mlx thinking per request ([864f6ac](https://github.com/michaelasper/kir-ai/commit/864f6ac97a1429f46e00b5e39b3017af61fafc16))
* distinguish qwen bf16 profile ([e79e855](https://github.com/michaelasper/kir-ai/commit/e79e855cfe58724957a0298caf52028ccf1f723c))
* emit sse heartbeats during backend stalls ([337c107](https://github.com/michaelasper/kir-ai/commit/337c107c35a7387702d5f1c2e69138de4ba77bc9))
* encode hub repo request paths ([6a898ad](https://github.com/michaelasper/kir-ai/commit/6a898adfd2cf9f4ef0569fb735ad9084418e38ee)), closes [#44](https://github.com/michaelasper/kir-ai/issues/44)
* execute qwen attention with required weights ([608dafe](https://github.com/michaelasper/kir-ai/commit/608dafe0f0ab5b88349b5b88bffc9bd6b2d1bc5a))
* fail closed for uncached native qwen multi-token decode ([b005401](https://github.com/michaelasper/kir-ai/commit/b005401983ee9288a92668768811aff844568821))
* group hub download inputs ([cd3cd41](https://github.com/michaelasper/kir-ai/commit/cd3cd416bc35248583b21240caa367f7c2dda4f7))
* harden hub snapshot artifact integrity ([5e71661](https://github.com/michaelasper/kir-ai/commit/5e716616d1aad177004a810bc78e3640e551809d))
* honor chat tool and streaming semantics ([d3b091c](https://github.com/michaelasper/kir-ai/commit/d3b091cdb3da59b0760240a36d75b1ea7dff64d6))
* include metadata in stream errors ([3e50842](https://github.com/michaelasper/kir-ai/commit/3e50842ebfc4a82efe7bdb0736eecbae4a65bef3)), closes [#42](https://github.com/michaelasper/kir-ai/issues/42)
* include profile in snapshot paths ([b6df02d](https://github.com/michaelasper/kir-ai/commit/b6df02dee349d437ea93e7aadcefe78cd22ccf9f))
* isolate model pull staging directories ([e8e51b3](https://github.com/michaelasper/kir-ai/commit/e8e51b3083c9267125c303bd6a23321dfb218065))
* preserve explicit generation limits ([dd34c95](https://github.com/michaelasper/kir-ai/commit/dd34c95df173082a17a89fbcebe6833a0963a2fc))
* preserve mlx tool calls and cancel stalled streams ([3ad2e5f](https://github.com/michaelasper/kir-ai/commit/3ad2e5fe5de8effe4b5b99a65dc5f7b5e66f7a37))
* protect admin endpoints with bearer auth ([33b3954](https://github.com/michaelasper/kir-ai/commit/33b395440bc6a20ac2010fce16f1d3ead2840dea))
* reject ChatML control tokens in prompts ([efd5537](https://github.com/michaelasper/kir-ai/commit/efd553713e053b10ad45d7124ee1d5b710e5b17f))
* reject required tool choice without tools ([e3ef5d4](https://github.com/michaelasper/kir-ai/commit/e3ef5d4727a77cd2b5b9c8ad45b50cd2ddd5fcc1))
* reject unsafe safetensors shard paths ([ded78eb](https://github.com/michaelasper/kir-ai/commit/ded78eb4ecbd88a1df9dbb1ded98f6450de51b04))
* reject unsupported choice counts ([3dc2083](https://github.com/michaelasper/kir-ai/commit/3dc208314c173921a551d1cb2a32b973a66c3920))
* reject unsupported completion sampling ([3991363](https://github.com/michaelasper/kir-ai/commit/3991363395e7721e2fe270c4694d2d6c19702f41))
* reject unsupported logprob controls ([eee90ab](https://github.com/michaelasper/kir-ai/commit/eee90ab1bf893a5269c90109c1b9be2722382a1c))
* reject unsupported parallel tool calls ([5c496d6](https://github.com/michaelasper/kir-ai/commit/5c496d6482fa33ec97e2649dbe465904e7e0c436))
* reject unsupported penalty controls ([bbb4da1](https://github.com/michaelasper/kir-ai/commit/bbb4da11b3ba52bf0640fcbc55c1272bb82b6d1d))
* reject unsupported sampling controls ([45bf64a](https://github.com/michaelasper/kir-ai/commit/45bf64a857e7268bcb8aab3f3708141760b00ab4))
* reject zero max tokens ([d49ee2b](https://github.com/michaelasper/kir-ai/commit/d49ee2bbc728ef8493385ff17be19b89175df4d7))
* replace placeholders and reuse metal kernels ([01f6ce4](https://github.com/michaelasper/kir-ai/commit/01f6ce4ecbee37788cca4ad88a0d2cb344524253))
* report stalled sse backend streams ([f0a5e5a](https://github.com/michaelasper/kir-ai/commit/f0a5e5a4c8e7d0eebf0164578f3bf61b21abb9f4))
* require backend cancellation support ([1822a58](https://github.com/michaelasper/kir-ai/commit/1822a58274fdf72e45df961b247dfb19fcb8dc7b)), closes [#43](https://github.com/michaelasper/kir-ai/issues/43)
* require explicit serve backend ([b31c1f5](https://github.com/michaelasper/kir-ai/commit/b31c1f5e6fc28ec6ba1e76ec671ddc247300d4fe))
* return stable errors for malformed json ([5d4b371](https://github.com/michaelasper/kir-ai/commit/5d4b37144fe42655e2e96bf8a5964167656c415a))
* route mlx chat tools through family contracts ([c3aab0b](https://github.com/michaelasper/kir-ai/commit/c3aab0bb9bbdaeebdfef867b2c2273241019e85e))
* start sse responses before generation completes ([bdc18ed](https://github.com/michaelasper/kir-ai/commit/bdc18ed6b4576ac647bf2e908943f1d21a55f381))
* stream backend chunks through sse ([9b54269](https://github.com/michaelasper/kir-ai/commit/9b542692d5ab64a8bf437301c6b4ca72dafcd11f))
* stream chat stop sequences incrementally ([93470ec](https://github.com/michaelasper/kir-ai/commit/93470ecf20067816f413c5ce3b2226fdaa598d10))
* stream completion stop sequences incrementally ([4be6ade](https://github.com/michaelasper/kir-ai/commit/4be6ade5b8fae67497d37c3ae4d6cf6367a5b46c))
* validate generated tool calls ([7a20446](https://github.com/michaelasper/kir-ai/commit/7a20446c9f8dc8c199305ec68807d806a1f7cb79))
* validate requests before streaming or scheduling ([dc0b86e](https://github.com/michaelasper/kir-ai/commit/dc0b86eb6a08d3e3983b6b2e215e3466304eaba9))

### Performance Improvements

* implement zero-allocation inference hot path ([88be7e4](https://github.com/michaelasper/kir-ai/commit/88be7e475902ea24eff466b0f9618ec0b81ef108))
