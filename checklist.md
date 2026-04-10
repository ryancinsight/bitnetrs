# bitnet.rs — Implementation Checklist

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Complete and tested |
| 🔧 | Implemented, needs more tests |
| ⏳ | In progress |
| ❌ | Not started |
| 🚫 | Blocked |

---

## Phase 1 — Foundation

### Workspace & Build System

- ✅ `Cargo.toml` workspace manifest with all 9 crates
- ✅ Workspace-level dependency declarations
- ✅ Release profile: `opt-level=3`, `lto=thin`, `codegen-units=1`
- ✅ Dev profile: `opt-level=1` for faster test compilation

---

## Phase 2 — `bitnet-core`

### Error Handling (`error.rs`)
- ✅ `BitNetError` enum with 9 variants
- ✅ `Result<T>` alias
- ✅ Constructor helpers: `backend()`, `shape()`, `weight()`, `quant()`, `config()`
- ✅ `thiserror`-derived `Display` for all variants
- ✅ `#[from] std::io::Error` impl

### Model Configuration (`config.rs`)
- ✅ `ModelConfig` struct with all 9 fields
- ✅ `bitnet_2b_config()` canonical constructor
- ✅ `ModelConfig::validate()` — checks all dimension invariants
- ✅ `ModelConfig::head_dim()` → `hidden_size / num_attention_heads`
- ✅ `ModelConfig::heads_per_group()` → `num_attention_heads / num_key_value_heads`
- ✅ `ModelConfig::q_dim()`, `kv_dim()`
- ✅ `GenerationConfig` struct
- ✅ `GenerationConfig::greedy()`, `chat_defaults()`, `creative()`
- ✅ `GenerationConfig::validate()`
- ✅ Tests: valid/invalid configs, dimension checks

### Tensor (`tensor/`)
- ✅ `Tensor<T>` struct with `Vec<T>` data and `[usize; 4]` shape
- ✅ `Tensor::zeros()`, `from_vec()`, `from_vec_1d()`, `from_vec_2d()`
- ✅ Strides computed from shape (C-contiguous / row-major)
- ✅ `flat_index()`, `get()`, `get_mut()`
- ✅ `row_slice()`, `reshape()`, `fill()`, `zero()`
- ✅ `Index<usize>` and `IndexMut<usize>` operators
- ✅ `TensorView<'_, T>` non-owning view
- ✅ `DType` enum: F32, F16, BF16, I8, U8, I2
- ✅ `DType::byte_size()`, `bit_size()`, `is_float()`, `is_integer()`
- ✅ `DType::storage_bytes()` — handles I2 sub-byte packing
- ✅ `DType::from_str()` — case-insensitive + aliases
- ✅ Tests: construction, accessors, reshape, round-trips

### Quantisation (`quant/`)

#### Ternary (`quant/ternary.rs`)
- ✅ `TernaryWeight { data, scale, rows, cols }` struct
- ✅ `TernaryWeight::new()` — validates shape, scale > 0, ternary values (debug)
- ✅ `TernaryWeight::new_unchecked()`
- ✅ `TernaryWeight::row()`, `get()`, `numel()`, `pack()`, `packed_bytes()`
- ✅ `pack_ternary()` — 2-bit/value, little-endian bit packing
- ✅ `unpack_ternary()` — inverse of pack_ternary
- ✅ Tests: encoding, round-trips, all-zero, all-one, all-neg-one, error cases

#### AbsMean (`quant/absmean.rs`)
- ✅ `absmean_quantize(weights: &[f32]) -> Result<(Vec<i8>, f32)>`
- ✅ `absmean_quantize_bf16(weights: &[bf16]) -> Result<(Vec<i8>, f32)>`
- ✅ `absmean_quantize_f16(weights: &[f16]) -> Result<(Vec<i8>, f32)>`
- ✅ `absmean_dequantize(quantised: &[i8], scale: f32) -> Result<Vec<f32>>`
- ✅ `absmean_quantize_grouped()` — per-group/per-tile variant
- ✅ `ABSMEAN_MIN = 1e-5` floor for scale
- ✅ Tests: positive/negative/mixed weights, all-zero, NaN/Inf rejection, scale positivity, ternary invariant, bf16 consistency, roundtrip error bound

#### AbsMax (`quant/absmax.rs`)
- ✅ `absmax_quantize_row(x: &[f32]) -> Result<(Vec<i8>, f32)>`
- ✅ `absmax_dequantize(quantised: &[i8], scale: f32) -> Result<Vec<f32>>`
- ✅ `absmax_quantize_batch(matrix, cols)` — per-row batch quantisation
- ✅ `ternary_dot_product_quantised()` — fused W·x with i32 accumulator
- ✅ `Q8_MAX = 127.0`, `ABSMAX_MIN = 1e-5`
- ✅ Tests: max element → 127, negative max → -127, all-zero, roundtrip, i8 range, end-to-end pipeline

### Backend Abstraction (`backend/`)
- ✅ `Backend` trait with 8 methods
- ✅ `Device` enum: Cpu/Gpu/Npu with fields
- ✅ `Device` Display, Default, convenience constructors
- ✅ Blanket `impl Backend for Arc<dyn Backend>`
- ✅ `rms_norm_f32()` standalone function
- ✅ `rope_cos_sin_table()` — pre-computed frequency tables
- ✅ `apply_rope_to_head()` — single-head rotation
- ✅ `softmax_f32()` — numerically stable
- ✅ `squared_relu_f32()` — in-place
- ✅ `elementwise_mul_f32()` — in-place
- ✅ `lm_head_matmul()` — unquantised f32 matmul
- ✅ Tests: all math ops with analytical expected values

---

## Phase 3 — `bitnet-cpu`

### GEMV (`gemv.rs`)
- ✅ `ternary_gemv_f32()` — Rayon parallel outer loop
- ✅ `dot_ternary_f32()` — single-row inner kernel
- ✅ `dot_ternary_i8()` — integer accumulator variant
- ✅ `ternary_gemv_quantised()` — W_q·x_q with combined scale
- ✅ Shape validation before dispatch
- ✅ Tests: 2×3 matrix, identity, negation, scale linearity, zero input, error cases, quantised vs f32 consistency

### Normalisation (`norm.rs`)
- ✅ `rms_norm()` — checked variant
- ✅ `rms_norm_unchecked()` — hot path
- ✅ `rms_norm_inplace()` — in-place variant
- ✅ Tests: unit weight, scaled weight, zero input, large input, sign preservation, single element, mathematical properties (scale invariance, squared-mean ≈ 1)

### RoPE (`rope.rs`)
- ✅ `RopeCache::new()` — pre-compute cos/sin tables
- ✅ `RopeCache::at(pos)` — position-indexed lookup
- ✅ `apply_rope()` — compute angles on-the-fly
- ✅ `apply_rope_cached()` — use pre-computed tables
- ✅ `compute_cos_sin()` private helper
- ✅ `apply_rope_to_head_slice()` private helper
- ✅ Tests: position-0 identity, norm preservation, cached vs uncached equality, relative-position invariance theorem, error cases

### Attention (`attention.rs`)
- ✅ `masked_attention()` — causal GQA scaled dot-product
- ✅ GQA group assignment: `kv_head = h / heads_per_group`
- ✅ Attention scale: `1 / sqrt(head_dim)`
- ✅ Numerically stable softmax over scores
- ✅ Output as convex combination of V vectors
- ✅ Tests: single-position identity, uniform scores → averaged values, dominant score, GQA head assignment, convex combination property, finite outputs, error cases, scale linearity, softmax shift invariance, 2B model dimensions smoke test

### Activation (`activation.rs`)
- ✅ `squared_relu()` — in-place ReLU²
- ✅ `squared_relu_into()` — non-aliasing variant
- ✅ `sqrelu_gate()` — fused sqrelu(gate) ⊙ up
- ✅ `softmax()` — numerically stable in-place
- ✅ `softmax_partial()` — partial-slice variant
- ✅ Tests: negative → 0, positive → squared, mixed, continuity at 0, softmax sums to 1, uniform distribution, shift invariance, monotone preservation, two-element sigmoid formula

### CpuBackend (`lib.rs`)
- ✅ `CpuBackend::new(threads)` — Rayon pool initialisation
- ✅ `CpuBackend::into_arc()`
- ✅ `impl Backend for CpuBackend` — all 8 methods
- ✅ `#[instrument]` tracing on all Backend methods
- ✅ Tests: all Backend methods via factory, Arc forwarding, mini-transformer-block smoke test

---

## Phase 4 — `bitnet-gpu`

### Context (`context.rs`)
- ✅ `create_wgpu_device(device_id)` — async adapter selection
- ✅ `create_wgpu_device_blocking()` — pollster wrapper
- ✅ Adapter priority sorting: discrete > integrated > virtual > cpu
- ✅ `AdapterInfo` struct with name/backend/type/vendor/device
- ✅ `compute_limits()` — request adequate buffer sizes
- ✅ Software adapter warning
- ✅ Tests: enumeration no-panic, out-of-range device_id error, AdapterInfo display

### Buffer (`buffer.rs`)
- ✅ `BufferUsage` enum: Storage, StorageReadWrite, Uniform, Staging, Upload
- ✅ `GpuBuffer::new()` — allocate uninitialised
- ✅ `GpuBuffer::from_data<T: Pod>()` — initialise from slice
- ✅ `upload_f32()`, `upload_i8()`, `upload_i8_as_i32()`, `upload_u8()`
- ✅ `upload_uniform<T: Pod>()` — parameter struct upload
- ✅ `download_f32()` — staging buffer readback
- ✅ `as_binding()`, `raw()` — wgpu integration helpers
- ✅ Tests: upload/download roundtrip, size mismatch error, zero size error, i8-as-i32 upload

### WGSL Shaders (`shaders/`)
- ✅ `gemv.wgsl` — ternary GEMV with 256-thread workgroup reduction
- ✅ `norm.wgsl` — RMSNorm with 2-phase reduction + broadcast
- ✅ `rope.wgsl` — RoPE with rope_q, rope_k, rope_qk entry points
- ✅ `attention.wgsl` — causal GQA with online max+sum softmax, MAX_SEQ_LEN=4096
- ✅ All shaders declare correct `@group(0) @binding(N)` entries
- ✅ All shaders have 16-byte aligned uniform structs

### Pipelines (`pipeline.rs`)
- ✅ `GpuPipelines::new()` — compile all 4 shaders
- ✅ `GpuPipelines` fields: gemv, norm, rope, attention + layouts
- ✅ `create_gemv/norm/rope/attention_bind_group_layout()` helpers
- ✅ `compile_pipeline()` private helper
- ✅ `storage_read_entry()`, `storage_rw_entry()`, `uniform_entry()` helpers
- ✅ Embedded shaders via `include_str!`
- ✅ Tests: all pipelines compile, shader source content assertions, binding type assertions

### GpuBackend (`lib.rs`)
- ✅ `GpuBackend::new(device_id)` — async
- ✅ `GpuBackend::new_blocking()` — pollster wrapper
- ✅ `GpuBackend::into_arc()`
- ✅ `dispatch_gemv()`, `dispatch_rms_norm()`, `dispatch_rope()`, `dispatch_attention()`
- ✅ `impl Backend for GpuBackend` — all methods with CPU fallback on GPU error
- ✅ `#[instrument]` tracing on all Backend methods
- ✅ Pod parameter structs: GemvParams, NormParams, RopeParams, AttnParams
- ✅ Tests: init, wrong shape errors, position-0 RoPE identity, single-position attention, all device-skip-if-no-GPU pattern

---

## Phase 5 — `bitnet-npu`

### Detection (`detect.rs`)
- ✅ `NpuVendor` enum: Intel/Amd/Qualcomm/Apple/Samsung/MediaTek/Unknown
- ✅ `NpuAdapterType` enum: DiscreteNpu/IntegratedNpu/Virtual/Software/Unknown
- ✅ `NpuInfo` struct with all fields + Display
- ✅ `detect_npu()` — returns best NPU candidate or None
- ✅ `detect_all_npus()` — returns all candidates sorted by priority
- ✅ `is_npu_adapter()` — keyword-based heuristic
- ✅ `is_npu_adapter_extended(name, extra_keywords)` — accepts caller-supplied keyword slice
- ✅ `classify_vendor()` — PCI ID + name fallback
- ✅ `classify_adapter_type()` — DeviceType mapping
- ✅ `NPU_NAME_KEYWORDS` — extended keyword list (`"apu"`, `"exynos"`, `"mediatek"`, `"samsung"` added)
- ✅ `SAMSUNG_NPU_KEYWORDS` and `MEDIATEK_NPU_KEYWORDS` — vendor-specific keyword constants
- ✅ `BITNET_NPU_ADAPTER` environment variable override — selects adapter by index, bypassing name heuristics
- ✅ Tests: vendor classification, adapter type mapping, detect no-panic, consistency, sort order, GPU names don't false-positive, known NPU names match, Samsung/MediaTek name detection, `BITNET_NPU_ADAPTER` env-var override, extended-keyword API

### NpuBackend (`lib.rs`)
- ✅ `NpuBackend::new(device_id)` — always succeeds (CPU fallback)
- ✅ `NpuBackend::is_using_npu()`, `npu_info()`
- ✅ `NpuBackend::into_arc()`
- ✅ `impl Backend for NpuBackend` — all 8 methods delegated to inner
- ✅ `#[instrument]` tracing on all Backend methods
- ✅ Cross-backend consistency test (NPU == CPU when no NPU)
- ✅ Mini-forward-pass smoke test
- ✅ Tests: all Backend methods, error cases, device name format

---

## Phase 6 — `bitnet-weights`

### SafeTensors (`safetensors.rs`)
- ✅ `TensorMeta` struct with dtype/shape/data_offsets
- ✅ `TensorMeta::numel()`, `byte_size()`
- ✅ `parse_safetensors_header()` — JSON + trailing-null handling
- ✅ `load_bf16_safetensors()` — BF16/F16/F32 → f32 HashMap
- ✅ `load_raw_safetensors()` — raw bytes + shape + dtype
- ✅ `load_safetensors_meta()` — header-only fast path
- ✅ `bf16_bytes_to_f32()`, `f16_bytes_to_f32()`, `f32_bytes_to_f32()`
- ✅ Non-finite value replacement with 0.0 + warning
- ✅ Tests: TensorMeta, header parsing, __metadata__ ignored, BF16/F16/F32/U8 conversion, multiple tensors, file not found, all-finite invariant

### HuggingFace Hub (`hf_hub.rs`)
- ✅ `download_model_from_hf()` — async multi-file download with SHA256 sidecar verification
- ✅ `download_model_from_hf_blocking()` — tokio runtime wrapper
- ✅ Cache hit detection (skip if file exists and non-empty)
- ✅ Temporary file + atomic rename on success
- ✅ `download_file_with_retry()` — up to 3 retries with exponential back-off
- ✅ `is_retryable_error()` — 429/503/connection vs 401/403/404
- ✅ Progress bars via `indicatif` (with content-length or spinner)
- ✅ `HF_TOKEN` environment variable for auth
- ✅ `build_client()` — reqwest with timeouts and user-agent
- ✅ `compute_sha256_hex(path)` — synchronous SHA256 over 64 KiB read chunks; returns lowercase hex
- ✅ `try_download_sha256(url, token)` — async `.sha256` sidecar fetch; `None` on 404 or any error (non-fatal)
- ✅ Tests: empty files → empty map, empty repo_id error, empty revision error, cache hit (local file), URL format, retry/non-retry classification, blocking variant, known-content SHA256, empty-file SHA256, nonexistent-file SHA256 error

### Loader (`loader.rs`)
- ✅ `LayerWeights` struct with all 11 fields
- ✅ `ModelWeights` struct: config + embed_tokens + layers + final_norm + lm_head
- ✅ `load_weights_from_bf16()` — full loading pipeline
- ✅ `load_layer_weights()` — per-layer assembly
- ✅ `require_tensor()` — lookup with element count validation
- ✅ `quantise_weight()` — absmean quantisation on load
- ✅ Weight name mapping: all 13 HuggingFace tensor key patterns
- ✅ Weight tying: `embed_tokens` and `lm_head` both `Arc<Vec<f32>>` sharing one allocation (zero duplication, −~1.3 GB for 2B model)
- ✅ Tests: tiny config round-trip, ternary invariants, shape verification, finite values, `Arc::ptr_eq` memory-sharing assertion, quantisation error bound, scale reflects magnitude

---

## Phase 7 — `bitnet-tokenizer`

### Tokenizer (`lib.rs`)
- ✅ `Tokenizer::llama3()` — cl100k_base initialisation
- ✅ `Tokenizer::encode()` — with optional BOS prepend
- ✅ `Tokenizer::decode()` — special token filtering
- ✅ `Tokenizer::decode_single()` — streaming single token
- ✅ `Tokenizer::decode_with_special_tokens()` — maps IDs 128000/128001/128006/128007/128009 to string literals; fallback `<|special:{id}|>`
- ✅ `Tokenizer::bos_token_id()` = 128000
- ✅ `Tokenizer::eos_token_id()` = 128009
- ✅ `Tokenizer::vocab_size()` = 128256
- ✅ `Tokenizer::apply_chat_template()` — LLaMA 3 Instruct format
- ✅ `Tokenizer::encode_chat()` — template + encode combined
- ✅ `ChatMessage` struct with role/content + constructors
- ✅ Special token constants: BOS/EOT/START_HEADER/END_HEADER/EOT_ID
- ✅ Tests: init, BOS/EOS IDs, encode empty, BOS prepend, decode roundtrip, Unicode, special token skipping, chat template format (EOT count, order, open assistant header), encode_chat, roundtrip property for ASCII cases, all five special-token ID round-trips, fallback format, mixed-sequence interleaving

---

## Phase 8 — `bitnet-model`

### KV Cache
- ✅ `KVCache::new()` — pre-allocate per layer
- ✅ `KVCache::store_kv()` — append or overwrite
- ✅ `KVCache::store_kv()` — reject `position >= max_seq`
- ✅ `KVCache::advance()` — increment filled_positions
- ✅ `KVCache::advance()` — enforce `filled_positions < max_seq`
- ✅ `KVCache::k_slice()`, `v_slice()` — position-bounded views
- ✅ `KVCache::clear()` — reset for new conversation
- ✅ `KVCache::len()`, `is_empty()`
- ✅ Tests: new is empty, kv_stride, clear, advance, max_seq overflow rejection, advance overflow guard, k_slice correctness, out-of-bounds error

### BitNetModel
- ✅ `BitNetModel::new()` — config validate + backend create + weights
- ✅ `BitNetModel::forward()` — full 30-layer autoregressive forward pass
- ✅ Token embedding lookup
- ✅ Pre-attention RMSNorm + Q/K/V ternary GEMV
- ✅ RoPE applied to Q and K
- ✅ KV cache store (append/overwrite via `KVCache::store_kv()`)
- ✅ KV cache max_seq bound check before forward append
- ✅ Causal GQA attention
- ✅ Attention sub-norm + O projection
- ✅ Attention residual connection
- ✅ Pre-FFN RMSNorm + Gate/Up GEMV
- ✅ Squared ReLU gate + element-wise multiply
- ✅ FFN sub-norm + Down projection
- ✅ FFN residual connection
- ✅ Final RMSNorm + LM head matmul (weight-tied)
- ✅ Scratch buffer reuse across layers
- ✅ KV cache position advancement
- ✅ Tests: single token logit count, finite logits, empty tokens error, exceeds max_pos error, exceeds kv_cache max_seq error, kv_cache advances, autoregressive decode step, new_kv_cache dimensions

### Device Factory (`device.rs`)
- ✅ `create_backend(device)` — dispatch to cpu/gpu/npu
- ✅ Tests: CPU always succeeds, GPU errors on invalid id, NPU always succeeds, device convenience constructors, Arc clone, Arc shared across threads

---

## Phase 9 — `bitnet-inference`

### Sampling
- ✅ `SamplingConfig` struct with 6 fields
- ✅ `SamplingConfig::greedy()`, `chat_defaults()`, `creative()`
- ✅ `SamplingConfig::validate()`
- ✅ `sample_next_token()` — full pipeline: penalty → temp → top-k → top-p → softmax → LCG sample
- ✅ LCG PRNG: Knuth TAOCP constants, seeded from config.seed + past_tokens.len()
- ✅ Greedy shortcut (top_k=1 → argmax)
- ✅ Tests: greedy picks argmax, token in valid range, repetition penalty, determinism, different seeds may differ, all-neg-infinity except one, top-k constraint, temp near zero → greedy, sampling stress test (100 steps no panic), top-k correctness theorem test

### InferenceEngine
- ✅ `InferenceEngine::new()` — resolve sibling `config.json` when present, then load weights + model + tokenizer + kv_cache
- ✅ `resolve_model_config()` — sibling `config.json` auto-detection with canonical 2B fallback
- ✅ `InferenceEngine::generate()` — prefill + decode loop with EOS check
- ✅ `InferenceEngine::generate_chat()` — chat template + generate
- ✅ `InferenceEngine::reset()` — KV cache clear
- ✅ `InferenceEngine::tokenizer()`, `context_length()`
- ✅ Tests: sibling config path resolution, config.json load path, fallback-to-2B path

### ChatPipeline
- ✅ `ChatPipeline::new()` — engine + history + system_prompt
- ✅ `ChatPipeline::chat()` — build messages + generate + append to history
- ✅ `ChatPipeline::reset_conversation()` — clear history + engine reset
- ✅ `ChatPipeline::history()`, `system_prompt()`, `set_system_prompt()`

---

## Phase 10 — `bitnet-cli`

### Download subcommand
- ✅ `--repo`, `--revision`, `--cache-dir`, `--extra-files` args
- ✅ Calls `download_model_from_hf` async
- ✅ Prints downloaded file paths with sizes
- ✅ Prints next-step hint for `--model` path

### Generate subcommand
- ✅ `--model`, `--prompt` required args
- ✅ `--device` (cpu/gpu/npu), `--threads`, `--gpu-id`
- ✅ `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`, `--seed`
- ✅ Model file existence check with helpful error message
- ✅ Prints prompt prefix then continuation
- ✅ Per-generation timing report (tok/s)

### Chat subcommand
- ✅ `--model`, `--device`, `--threads`, `--gpu-id`
- ✅ `--system-prompt`, `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`
- ✅ `--no-labels` flag
- ✅ Welcome banner
- ✅ `/reset` command handler
- ✅ `exit`/`quit`/`Ctrl+D` exit handlers
- ✅ Per-turn error recovery with conversation reset

### CLI Tests
- ✅ `format_bytes()` — GiB/MiB/KiB/B formatting
- ✅ `resolve_cache_dir()` — arg > env > default
- ✅ `DeviceArg::into_device()` — all 3 variants
- ✅ All 3 subcommands parse with required args
- ✅ All 3 subcommands parse with all optional args
- ✅ Missing required args return parse error
- ✅ Unknown subcommand returns parse error
- ✅ `--version` and `--help` don't panic

---

## Phase 11 — Documentation

### In-Code Documentation
- ✅ Module-level `//!` doc comments in all files
- ✅ Mathematical invariants documented on all public types
- ✅ Algorithm derivations in function-level docs
- ✅ `# Errors`, `# Panics`, `# Example` sections on key functions
- ✅ `# Invariants` sections on structs with non-trivial contracts

### Project Artifacts
- ✅ `README.md` — comprehensive (architecture, quick-start, usage, perf, crate map, math)
- ✅ `backlog.md` — sprint backlog with all epics and stories
- ✅ `checklist.md` — this file
- ✅ `gap_audit.md` — known gaps and future work

---

## Quality Gates

| Gate | Status |
|------|--------|
| All public API has Rustdoc | ✅ |
| No `unwrap()` in library code (only `debug_assert`) | ✅ |
| No `todo!()` / `unimplemented!()` / `panic!()` in library code | ✅ |
| All error paths return `Result` | ✅ |
| `#[forbid(unsafe_code)]` on all library crates | ✅ |
| Every test asserts on computed VALUES (not just `is_ok()`) | ✅ |
| Test data derived from analytical solutions | ✅ |
| Negative tests for all error variants | ✅ |
| Boundary tests (empty slices, max values, position 0) | ✅ |
| No mock implementations (all tests use real computation) | ✅ |

---

## Known Limitations (see gap_audit.md)

- GPU backend allocates/uploads buffers per forward call (not persistent)
- No SIMD intrinsics for CPU GEMV (relies on auto-vectorisation)
- Tokenizer uses cl100k_base, not exact LLaMA 3 BPE vocab
- No streaming token output (waits for full generation)
- Single-batch inference only (no batched prefill)
- NPU detection covers Intel/AMD/Qualcomm/Apple/Samsung/MediaTek; Snapdragon `"Adreno"` adapters without an `"npu"` substring may still require the `BITNET_NPU_ADAPTER` env-var override
- Model config auto-detects from sibling `config.json`; fallback remains canonical 2B when absent
- ~~Golden output tests against the real 2B weights are pending~~ **Resolved**: weight scale bug (`.recip()` inversion) fixed; activation quantisation implemented via `ternary_gemv_with_activation_quant`; the `chat_generation` integration test now asserts `output.contains("Paris")` against real 2B weights, serving as an end-to-end golden output test (see G06 in gap_audit.md)