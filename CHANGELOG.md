# Changelog

All notable changes to this project will be documented in this file.

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
