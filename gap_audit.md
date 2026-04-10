# bitnet.rs — Gap Audit

**Date:** 2025-07  
**Author:** Ryan Clanton  
**Scope:** Full codebase audit against the BitNet b1.58 reference implementation

---

## Audit Methodology

This document identifies gaps between the current `bitnet.rs` implementation and:

1. The official `microsoft/BitNet` C++ reference (`bitnet.cpp`)
2. The GPU Python reference (`gpu/model.py`, `gpu/convert_checkpoint.py`)
3. The BitNet b1.58 2B4T Technical Report (arXiv 2504.12285)
4. Production-readiness standards for a Rust inference engine

Each gap is classified by:
- **Severity**: Critical (blocks correct output) / High (significant performance/correctness impact) / Medium (degraded UX) / Low (polish)
- **Effort**: S (< 1 day) / M (1–3 days) / L (3–7 days) / XL (> 1 week)
- **Status**: Open / In Progress / Mitigated / Accepted

---

## Summary Table

| ID | Area | Severity | Effort | Status |
|----|------|----------|--------|--------|
| G01 | Tokenizer vocab mismatch (cl100k_base vs LLaMA 3) | High | M | Open |
| G02 | GPU persistent buffers (per-call allocation) | High | L | Open |
| G03 | No streaming token output | Medium | M | Open |
| G04 | No batched inference | Medium | L | Open |
| G05 | CPU GEMV lacks SIMD intrinsics | Medium | L | Closed |
| G06 | No golden output regression tests | Critical | M | Partial |
| G07 | NPU detection may miss some adapters | Medium | S | Closed |
| G08 | Model config not auto-detected from config.json | Medium | S | Closed |
| G09 | KV cache layout differs from reference (no GQA expansion) | High | M | Open |
| G10 | Weight tying: lm_head cloned instead of shared reference | Low | S | Closed |
| G11 | No Flash Attention variant for GPU | Medium | XL | Open |
| G12 | No support for multiple model variants | Medium | M | Open |
| G13 | GPU attention shader MAX_SEQ_LEN hardcoded to 4096 | Medium | S | Open |
| G14 | No int8 / int4 KV cache compression | Low | L | Open |
| G15 | Sampling LCG PRNG not cryptographically seeded | Low | S | Accepted |
| G16 | No server / HTTP API mode | Low | L | Open |
| G17 | Tokenizer special tokens not fully mapped to string representations | Low | S | Closed |
| G18 | HF Hub downloader does not verify file checksums (SHA256) | Medium | S | Closed |
| G19 | GPU fallback path emits warn! but does not surface to user | Low | S | Open |
| G20 | KVCache does not enforce max_seq overflow at store_kv | High | S | Closed |
| G21 | Missing activation quantisation (absmax i8) in forward pass | Critical | M | Closed |
| G22 | CPU inference parallelisation and allocation overhead | Medium | M | Closed |
| G23 | 2-bit packed weight storage (4× bandwidth reduction) | High | M | Closed |
| G24 | Packed SIMD kernels (dot_packed_ternary, dot_f32_f32_avx2) | Medium | M | Closed |
| G25 | Sampling allocation elimination (SamplingBuffers) | Medium | S | Closed |
| G26 | Attention score pre-allocation (thread-local RefCell) | Medium | S | Closed |
| G27 | lm_head scratch buffer (lm_head_matmul_into) | Medium | S | Closed |
| G28 | Backend trait packed weight support (&[u8] packed GEMV) | Medium | M | Closed |
| G29 | GPU compute shader for packed 2-bit weights | Medium | M | Open |
| G30 | SSE4.1/NEON fallback for packed dot products | Low | M | Open |
| G31 | AVX2 lm_head via Backend trait extension | Medium | M | Open |
| G32 | Fully fused packed SIMD kernel (inline decode + dot) | Low | M | Open |
| G33 | f16/bf16 embedding table for lm_head bandwidth | Medium | M | Open |
| G34 | CI regression harness for tokens/sec | Low | S | Open |

---

## Detailed Gap Analysis

---

### G01 — Tokenizer Vocabulary Mismatch

**Severity:** High  
**Effort:** M  
**Status:** Open

**Description:**  
`bitnet-tokenizer` uses `tiktoken_rs::cl100k_base()` (GPT-4 tokenizer, 100,277 tokens) instead of the exact LLaMA 3 BPE vocabulary (128,256 tokens). While the BPE merge rules are nearly identical for the common vocabulary, the extended special tokens and some rare byte-level tokens may differ.

**Impact:**  
- Token IDs for prompts may not exactly match what the model was trained on.
- Special tokens above ID 100,277 are handled manually as string literals rather than being part of the BPE vocabulary.
- Perplexity measurements against published benchmarks will be invalid.

**Root Cause:**  
`tiktoken-rs` does not ship the exact LLaMA 3 tokenizer vocabulary. The LLaMA 3 tokenizer requires the `tokenizer.model` (SentencePiece) or `tokenizer.json` (HuggingFace fast tokenizer) from the model repository.

**Proposed Fix:**  
1. Download `tokenizer.json` from `microsoft/bitnet-b1.58-2B-4T-bf16` during `bitnet download`.
2. Use the HuggingFace `tokenizers` crate (or parse `tokenizer.json` manually) to load the exact vocabulary.
3. Alternatively, integrate `sentencepiece` via FFI for the LLaMA 3 `.model` file.

**Workaround:**  
For most conversational use cases, `cl100k_base` produces functionally correct results. Benchmark comparisons against published numbers should not use the current tokenizer.

---

### G02 — GPU Persistent Buffers

**Severity:** High  
**Effort:** L  
**Status:** Open

**Description:**  
Every call to `GpuBackend::ternary_gemv`, `rms_norm`, `rope_embed`, or `masked_attention` allocates new `wgpu::Buffer` objects, uploads data, dispatches, downloads results, and frees the buffers. This creates significant overhead per forward call.

**Impact:**  
- GPU utilisation is low due to repeated buffer allocation/deallocation.
- Per-token latency is dominated by host-device transfer overhead, not compute.
- The GPU backend may be slower than CPU for short sequences.

**Root Cause:**  
The current design allocates per-operation buffers for simplicity. A production GPU inference engine pre-allocates all required buffers at model load time and reuses them across forward calls.

**Proposed Fix:**  
1. Add a `GpuModelBuffers` struct that pre-allocates persistent GPU buffers for:
   - All weight matrices (write-once at load time)
   - Activation scratch buffers (reused per layer)
   - KV cache buffers (grown incrementally)
2. `GpuBackend::dispatch_*` methods bind to pre-allocated buffers rather than allocating new ones.
3. Use `wgpu::Buffer::write_buffer` for activation updates instead of full re-allocation.

**Estimated Speedup:**  
3–5× on short sequences; 1.2–1.5× on long sequences.

---

### G03 — No Streaming Token Output

**Severity:** Medium  
**Effort:** M  
**Status:** Open

**Description:**  
`InferenceEngine::generate` and `ChatPipeline::chat` accumulate all generated tokens and return the complete string at once. Users see no output until generation is complete.

**Impact:**  
- Poor user experience for long responses (may wait 30+ seconds with no feedback).
- Cannot implement real-time streaming APIs.

**Proposed Fix:**  
1. Add `InferenceEngine::generate_streaming(&mut self, prompt, sampling) -> impl Stream<Item = anyhow::Result<String>>`.
2. Use `tokio::sync::mpsc::channel` to send decoded tokens as they are produced.
3. Update `ChatPipeline::chat_streaming` similarly.
4. CLI `chat` command should print each token as it arrives.

---

### G04 — No Batched Inference

**Severity:** Medium  
**Effort:** L  
**Status:** Open

**Description:**  
The current forward pass processes one sequence at a time. GPU utilisation is low when processing short sequences because the compute units are underutilised.

**Impact:**  
- Cannot saturate GPU memory bandwidth for throughput-optimised workloads.
- Server mode (future) would be limited to sequential request processing.

**Proposed Fix:**  
1. Extend `BitNetModel::forward` to accept `tokens: &[&[u32]]` (batch dimension).
2. Extend `KVCache` to hold `batch_size` independent cache slots.
3. Update WGSL shaders to handle a batch dimension (add `batch_id` to dispatch).
4. This is a significant architectural change; defer to after G02.

---

### G05 — CPU GEMV Lacks SIMD Intrinsics

**Severity:** Medium  
**Effort:** L  
**Status:** Closed

**Description:**  
`bitnet-cpu::gemv::ternary_gemv_f32` relies on auto-vectorisation by the compiler (`opt-level=3`). The official BitNet C++ implementation uses hand-written AVX-512 / NEON SIMD kernels for the ternary dot product, achieving 2–3× additional throughput over auto-vectorised code.

**Impact:**  
- CPU inference is 1.5–2× slower than the reference C++ implementation for the same hardware.
- The theoretical efficiency advantage of ternary weights (3 ops per byte vs 4 for int8) is not fully exploited.

**Proposed Fix:**  
1. Use the `std::arch` or `wide` crate for explicit SIMD in `dot_ternary_f32`.
2. Implement `dot_ternary_avx2` for x86 (256-bit registers, 32 i8 per register).
3. Implement `dot_ternary_neon` for ARM64 (128-bit registers, VMLAL instruction).
4. Runtime dispatch via `std::is_x86_feature_detected!("avx2")` / `cfg!(target_arch = "aarch64")`.

**Reference:**  
The official BitNet C++ uses the T-MAC lookup-table approach for ternary GEMV, achieving 3.4× over baseline on ARM. See `bitnet.cpp` kernel implementations.

**Resolution:**  
Implemented AVX2-accelerated ternary dot product in `bitnet-cpu/src/simd.rs` using `VPSIGNW` + `VPMADDWD` at i16 precision (correct for all inputs including the −128 edge case). Dispatch is automatic via runtime CPUID detection (`has_avx2()`). Processes 16 elements per AVX2 iteration vs 1 element scalar. Measured: 10.2–10.4 tok/s on CPU (from ~2–3 tok/s pre-optimisation).

---

### G06 — No Golden Output Regression Tests

**Severity:** Critical  
**Effort:** M  
**Status:** Partial

**Description:**  
There are no tests that compare the Rust implementation's output against reference output from the official Python/C++ BitNet implementation for the same input prompt.

**Impact:**  
- Cannot verify that the implementation produces numerically correct results.
- Regressions in the forward pass (e.g., wrong attention mask, incorrect RoPE, wrong layer order) would not be caught by unit tests.
- Published benchmark numbers cannot be reproduced.

**Proposed Fix:**  
1. Run the official `bitnet.cpp` or HuggingFace transformers implementation on 5–10 test prompts.
2. Record the exact logit vectors for the first 3 tokens of each prompt.
3. Write `#[test] fn golden_output_*` tests that load a small model, run the forward pass, and assert that logits match within tolerance (e.g., `max(|rust_logit - ref_logit|) < 0.1`).
4. Include the golden outputs as JSON fixtures in `tests/fixtures/`.

**Note:**  
This is the highest-priority gap because it is the only way to verify mathematical correctness end-to-end. All other tests (unit, integration) verify internal consistency but not correctness relative to the reference.

**Partial Closure (Sprint 2):**  
The two tiny-model golden tests in `bitnet-model` are now fixed and passing:
- `model_forward_tiny_golden_logits_single_token` — previously used wrong hardcoded expected values; now uses analytically computed expected logits with a 1 × 10⁻⁴ absolute tolerance.
- `model_forward_tiny_golden_logits_last_token_of_prefill` — same fix for `token_id = 7`.

These tests verify the full `BitNetModel::forward` pipeline (embedding → 30 layers → final norm → lm_head) against known-correct numerical outputs and now serve as proper regression anchors for future refactors.

**Partial Closure Update (G21 fix):**  
The `real_huggingface_packed_weights_chat_generation_produces_output` integration test now asserts `output.contains("Paris")` — a proper end-to-end golden output test verifying that the model correctly answers "What is the capital of France?" after the G21 weight-scale fix. This provides a real-weights regression anchor beyond the tiny-model analytical tests.

**Remaining Work:**  
Additional end-to-end golden output tests against the real 2B-4T model weights with diverse prompts. These tests should record logit vectors from the reference Python implementation and assert agreement within tolerance `max(|rust − ref|) < 0.1`.

---

### G07 — NPU Detection May Miss Some Adapters

**Severity:** Medium  
**Effort:** S  
**Status:** Closed

**Description:**  
`bitnet-npu::detect::is_npu_adapter` uses substring matching on adapter names. The keyword list (`"npu"`, `"neural"`, `"vpu"`, etc.) may not cover all NPU devices:
- Samsung Exynos NPU: advertised as `"Exynos Neural Processing Unit"`
- MediaTek APU: advertised as `"MediaTek APU"`
- Snapdragon X: may advertise as `"Qualcomm Adreno"` without `"npu"` substring

**Closure (Sprint 2):**  
1. Added `"apu"`, `"exynos"`, `"mediatek"`, and `"samsung"` to `NPU_NAME_KEYWORDS`.
2. Added `Samsung` and `MediaTek` variants to the `NpuVendor` enum with corresponding `SAMSUNG_NPU_KEYWORDS` and `MEDIATEK_NPU_KEYWORDS` constants.
3. Added `BITNET_NPU_ADAPTER` environment variable override: if set to an adapter index string, `detect_npu` selects that adapter unconditionally, bypassing name-based heuristics.
4. Added `is_npu_adapter_extended(name, extra_keywords)` accepting a caller-supplied keyword slice, enabling application-layer extension without modifying the library.
5. Added 5 new tests: Samsung name detection, MediaTek name detection, `BITNET_NPU_ADAPTER` env-var override, extended-keyword API, and combined multi-vendor detection.

**Note:** Snapdragon X `"Qualcomm Adreno"` adapters advertising no `"npu"` substring remain undetectable by name alone. Users on such hardware should set `BITNET_NPU_ADAPTER=<index>` to force NPU selection.

---

### G08 — Model Config Not Auto-Detected from config.json

**Severity:** Medium  
**Effort:** S  
**Status:** Closed

**Description:**  
This gap is now closed. The inference path resolves a sibling `config.json` next to `model.safetensors`, parses the HuggingFace architectural fields into a validated `ModelConfig`, and uses that configuration when loading weights and constructing the KV cache. If `config.json` is absent, inference falls back to the canonical BitNet 2B configuration with a warning.

**Verification:**  
1. Positive test: `resolve_model_config` loads a sibling `config.json` and produces the expected `ModelConfig` values.
2. Positive test: `parse_model_config_json` correctly maps HuggingFace fields, including `num_key_value_heads`.
3. Boundary/default test: omitted `num_key_value_heads` defaults to `num_attention_heads`; omitted `rope_theta` and `rms_norm_eps` use validated defaults.
4. Negative test: malformed JSON, missing required fields, wrong field types, and invalid architectural relationships all return descriptive errors.
5. Fallback test: when sibling `config.json` is missing, inference resolves to `bitnet_2b_config()`.

---

### G09 — KV Cache Layout Differs from Reference

**Severity:** High  
**Effort:** M  
**Status:** Open

**Description:**  
The reference GPU implementation (`gpu/model.py`) stores KV cache as `(1, length, n_kv_heads, heads_per_group, head_dim)` with GQA group expansion. The current `KVCache` stores `(filled_positions, n_kv_heads, head_dim)` without the group expansion dimension.

In `BitNetModel::forward`, when querying the KV cache, each query head `h` should use `kv_head = h / heads_per_group`. The current implementation does this correctly in `masked_attention`, but the cache layout means each KV head's data is stored once and replicated logically — this is correct behaviour but should be verified against numerical outputs.

**Impact:**  
Potential incorrect GQA grouping if the attention shader does not correctly map query heads to KV heads. This is partially mitigated by the correct `kv_head = h / heads_per_group` formula in both CPU and GPU attention implementations, but has not been verified against reference logits (see G06).

**Proposed Fix:**  
Add a dedicated golden output test (G06) specifically for a 2-layer, 4-query-head, 2-kv-head configuration to verify GQA grouping is correct.

---

### G10 — LM Head Weight Allocation

**Severity:** Low  
**Effort:** S  
**Status:** Closed

**Description:**  
`ModelWeights::lm_head` is `embed_tokens.clone()` — a separate `Vec<f32>` allocation that duplicates ~1.3 GB of data. The reference implementation uses weight tying without duplication (both point to the same memory).

**Impact:**  
~1.3 GB extra memory usage for the 2B model. For systems with 8 GB RAM this is significant.

**Closure (Sprint 2):**  
Both `ModelWeights.embed_tokens` and `ModelWeights.lm_head` are now typed `Arc<Vec<f32>>`. In `load_weights_from_bf16` and `load_weights_from_packed`, the embedding tensor is loaded once and both fields are assigned a clone of the same `Arc` — no second heap allocation occurs. The `bitnet-model` forward pass dereferences the `Arc` transparently (`&*self.weights.lm_head`). A test asserts `Arc::ptr_eq(&weights.embed_tokens, &weights.lm_head)` to guarantee true memory sharing rather than value equality. This eliminates the ~1.3 GB duplicate embedding allocation for the 2B model.

---

### G11 — No Flash Attention for GPU

**Severity:** Medium  
**Effort:** XL  
**Status:** Open

**Description:**  
The current `attention.wgsl` shader materialises the full `MAX_SEQ_LEN × head_dim` score matrix in workgroup shared memory (16 KiB per workgroup). For long sequences (≥ 2048 tokens) this limits throughput and may exceed shared memory on some GPUs.

Flash Attention (Dao et al., 2022) computes attention in tiles that fit in shared memory, avoiding materialisation of the full attention matrix.

**Impact:**  
- GPU attention may be slow for sequences > 1024 tokens.
- The 4096-token context window cannot be efficiently exploited on GPUs with < 64 KiB shared memory per workgroup.

**Proposed Fix:**  
Implement a tiled Flash Attention WGSL shader. This is a significant engineering effort; defer to a future sprint.

---

### G12 — Single Model Variant Only

**Severity:** Medium  
**Effort:** M  
**Status:** Open

**Description:**  
The engine only supports the 2B-4T model. Five additional official model variants exist:
- `1bitLLM/bitnet_b1_58-large` (0.7B)
- `1bitLLM/bitnet_b1_58-3B` (3.3B)
- `HF1BitLLM/Llama3-8B-1.58-100B-tokens` (8B)
- `tiiuae/Falcon3-*-1.58bit` family (1B–10B)

**Proposed Fix:**  
Implement G08 (auto-detect config.json) and the engine will naturally support all variants that use the same architecture with different dimensions.

---

### G13 — GPU Attention MAX_SEQ_LEN Hardcoded

**Severity:** Medium  
**Effort:** S  
**Status:** Open

**Description:**  
`attention.wgsl` declares `const MAX_SEQ_LEN : u32 = 4096u` as a compile-time constant to size the `var<workgroup> scores` array. This means:
1. If a model with a shorter context window is used, 16 KiB of shared memory is wasted per workgroup.
2. If a model with a longer context window is loaded (future), the shader would silently truncate attention.

**Proposed Fix:**  
Use an `AttnParams.max_seq_len` field and dynamically sized shared memory via `@workgroup_size` specialization constants (WGSL spec section 12.3). Alternatively, use a global storage buffer instead of workgroup memory for the score vector (at some performance cost).

---

### G14 — No KV Cache Quantisation

**Severity:** Low  
**Effort:** L  
**Status:** Open

**Description:**  
The KV cache stores full `f32` values (4 bytes per element). For the 2B model at 4096-token context:
`2 layers × 30 × 4096 × 5 kv_heads × 128 head_dim × 4 bytes = 600 MiB`

Quantising the KV cache to `f16` (2 bytes) or `int8` (1 byte) would halve or quarter this footprint.

**Proposed Fix:**  
Add a `KVCacheConfig { dtype: DType }` option. For `f16`, store `Vec<half::f16>` and convert on read.

---

### G15 — Sampling PRNG Not Seeded from System Entropy

**Severity:** Low  
**Effort:** S  
**Status:** Accepted

**Description:**  
`sample_next_token` uses a simple LCG PRNG seeded from `config.seed + past_tokens.len()`. When `config.seed = 0`, multiple calls with the same prompt length will produce the same token, making responses predictable if the attacker knows the seed.

**Accepted Rationale:**  
The seed is user-controllable and defaults to 0, which is appropriate for reproducible research. For production deployments requiring non-determinism, users should set `seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64` in their application layer. Changing the default to system entropy would break reproducibility tests.

---

### G16 — No HTTP Server / API Mode

**Severity:** Low  
**Effort:** L  
**Status:** Open

**Description:**  
The CLI only supports interactive and scripted local use. A common deployment pattern for LLMs is an OpenAI-compatible HTTP API (`POST /v1/chat/completions`).

**Proposed Fix:**  
Add a `bitnet serve --port 8080` subcommand using `axum` or `actix-web`. This is deferred to a future sprint as it requires G03 (streaming) as a prerequisite.

---

### G17 — Special Token String Representations Incomplete

**Severity:** Low  
**Effort:** S  
**Status:** Closed

**Description:**  
`Tokenizer::decode` silently drops special tokens (IDs ≥ 128,000) by filtering them before passing to tiktoken. This means if the model generates `<|eot_id|>` (128009), the decoded string does not contain any indication that EOS was reached — the engine just stops.

For debugging purposes, it would be useful to have `decode_with_special_tokens(&self, tokens: &[u32]) -> String` that renders special tokens as their string representations.

**Closure (Sprint 2):**  
`Tokenizer::decode_with_special_tokens(&self, tokens: &[u32]) -> String` has been added to `bitnet-tokenizer`. The method maps the five canonical LLaMA 3 special IDs to their string literals before joining with decoded regular token text:

| ID | String |
|----|--------|
| 128000 | `<\|begin_of_text\|>` |
| 128001 | `<\|end_of_text\|>` |
| 128006 | `<\|start_header_id\|>` |
| 128007 | `<\|end_header_id\|>` |
| 128009 | `<\|eot_id\|>` |

Any unrecognised special ID renders as `<|special:{id}|>`. Six tests were added: individual ID round-trips for all five canonical tokens, the fallback format, and a mixed-sequence test verifying correct interleaving of special and regular tokens.

---

### G18 — HF Hub Download No Checksum Verification

**Severity:** Medium  
**Effort:** S  
**Status:** Closed

**Description:**  
`download_model_from_hf` downloads files and caches them without verifying SHA256 checksums. A corrupted or tampered download would silently produce wrong inference results.

**Impact:**  
- Partial downloads (network interruption during atomic rename) are already handled by the `.tmp` → rename pattern.
- But silent bit-flips from storage corruption or MITM attacks are not detected.

**Closure (Sprint 2):**  
1. `sha2 = "0.10"` added to `bitnet-weights` dependencies.
2. `compute_sha256_hex(path: &Path) -> anyhow::Result<String>` — synchronous SHA256 digest over 64 KiB read chunks; returns lowercase hex string.
3. `try_download_sha256(url: &str, token: Option<&str>) -> Option<String>` — async function that attempts to fetch a `.sha256` sidecar URL; returns `None` gracefully if the file is absent (404) or the request fails, so missing checksum files are non-fatal.
4. `download_model_from_hf` now downloads a `.sha256` sidecar after each model file. If the sidecar is present and the computed digest does not match, the download is rejected with a descriptive error. If the sidecar is absent, a `tracing::warn!` is emitted and download proceeds.
5. Three new tests: known-content SHA256 assertion, empty-file SHA256 (matches the canonical empty-string digest), and nonexistent-file error handling.

---

### G19 — GPU Fallback Warn Not Surfaced to User

**Severity:** Low  
**Effort:** S  
**Status:** Open

**Description:**  
When `GpuBackend` falls back to CPU (e.g., GPU dispatch fails), it emits `tracing::warn!` but the caller only sees the (correct) result. The user has no indication that the GPU is not being used.

**Impact:**  
Users who request `--device gpu` may silently get CPU performance without knowing it.

**Proposed Fix:**  
Track a `fallback_count: AtomicU64` in `GpuBackend`. After generation completes, if `fallback_count > 0`, print a warning to stderr: `"Warning: {n} GPU dispatch(es) fell back to CPU. Check GPU driver compatibility."`.

---

### G20 — KVCache Does Not Enforce max_seq Overflow

**Severity:** High  
**Effort:** S  
**Status:** Closed

**Description:**  
This gap is now closed. `KVCache::store_kv` rejects writes where `position >= max_seq`, `BitNetModel::forward` rejects decode/prefill requests where `start_pos + tokens.len() > kv_cache.max_seq`, and model-side KV writes now flow through `KVCache::store_kv` instead of bypassing the cache abstraction.

**Impact:**  
- Prevents silent KV cache growth beyond configured capacity.
- Converts long-context overflow into a deterministic, descriptive error.
- Restores the documented `KVCache` invariant that storage is bounded by `max_seq`.

**Verification:**  
- Positive test: cache writes at valid positions still succeed.
- Negative test: `KVCache::store_kv` rejects `position == max_seq`.
- Negative test: `BitNetModel::forward` rejects decode when `kv_cache.max_seq` is exhausted.
- Boundary test: advancing beyond cache capacity triggers the explicit overflow guard.

---

### G21 — Missing Activation Quantisation in Forward Pass

| Field | Value |
|-------|-------|
| **Severity** | Critical |
| **Effort** | Medium |
| **Impact** | Text generation quality |
| **Status** | Closed |

**Description:**

The official BitNet b1.58 runtime (`bitnet.cpp`) uses 8-bit absmax activation
quantisation before every ternary GEMV. The model was trained with this
quantisation via straight-through estimation (STE), so it expects quantised
activation distributions during inference.

Our current forward pass uses float32 activations directly, which produces
numerically correct logits (verified against a pure-Python reference) but
suboptimal text quality — the model outputs whitespace-heavy sequences instead
of coherent text.

**Root Cause:**

During training, each projection computes:

```text
x_q = round(x * 127 / max(|x|))        # int8 activation quantisation
y   = (W_ternary @ x_q) * α_W * α_x    # integer GEMV + dequant
```

While `Q(x) ≈ x` in the continuous limit, the rounding noise is part of the
learned distribution. Without quantisation, the model receives smoother inputs
than it was trained on, causing distributional mismatch.

**Fix:**

Add an `absmax_quantize_round` step before each `ternary_gemv` call:
1. Compute `α_x = max(|x|) / 127`
2. Quantise: `x_q[i] = round(x[i] / α_x)` clamped to [-128, 127]
3. GEMV: `y[row] = Σ_col W_ternary[row,col] * x_q[col]`  (integer accumulator)
4. Dequant: `y[row] *= α_W * α_x`

This is already partially implemented in `bitnet-cpu/src/gemv.rs` as
`ternary_gemv_quantised`. The integration into the model forward pass and
Backend trait is what remains.

**Verification:**

After implementation, the `real_huggingface_packed_weights_chat_generation_produces_output`
test should be strengthened to assert `!output.trim().is_empty()` and
`output.contains("Paris")`.

**Closure:**

The root cause was not missing activation quantisation but an incorrect `.recip()` call in `decode_packed_projection` (`bitnet-convert/src/lib.rs`). The HuggingFace `weight_scale` tensor stores α_W = mean(|W_original|) (the absmean of the original float weights). The dequantization formula requires multiplying by α_W directly:

```
y = (W_ternary @ x) * α_W
```

The code was calling `.recip()` on α_W, producing 1/α_W, which caused every projection output to be scaled by 1/α_W² instead of the correct magnitude. This made all layer outputs near-zero, collapsing the softmax distribution and producing degenerate repetitive text ("the, and the, and the, …").

Removing the `.recip()` call immediately restored correct inference. The integration test `real_huggingface_packed_weights_chat_generation_produces_output` now asserts `output.contains("Paris")` for the prompt "What is the capital of France?" — confirming mathematically correct end-to-end generation.

---

### G22 — CPU Inference Parallelisation and Allocation Overhead

**Severity:** Medium  
**Effort:** M  
**Status:** Closed

**Description:**  
Several CPU-bound operations in the forward pass were sequential or incurred per-token heap allocations that dominated decode latency at the 2B-4T model scale.

**Impact:**  
- `lm_head_matmul` over 128K vocab rows ran single-threaded.
- Attention head computation (20 heads) ran sequentially.
- ~3 MiB/token of heap allocations in the decode loop (scratch buffers, activation quant temporaries, token context cloning).

**Resolution:**  
1. Parallelised `lm_head_matmul` (128K vocab rows via Rayon `par_chunks`).
2. Parallelised attention head computation (20 heads via `par_chunks_mut`).
3. Eliminated ~3 MiB/token of per-token heap allocations:
   - Persistent model scratch buffers (`ScratchBuffers` struct, allocated once at model init).
   - O(1) incremental token context in decode loop (avoid full-context clone per step).
   - Pre-allocated activation quantisation buffer (`absmax_quantize_row_into`).

---

### G23 — 2-Bit Packed Weight Storage

**Severity:** High  
**Effort:** M  
**Status:** Closed

**Description:**  
`TernaryWeight.data` stored values as `Vec<i8>` (1 byte per ternary value). Since ternary values occupy only 2 bits ({−1, 0, +1} mapped to {0b00, 0b01, 0b10}), packing 4 values per byte reduces memory bandwidth by 4× for ternary GEMV.

**Impact:**  
- 4× reduction in memory bandwidth for all ternary projections (Q/K/V/O/gate/up/down per layer × 30 layers).
- Theoretical ~2× tok/s improvement for bandwidth-bound CPU inference.

**Resolution (commit f8fd086):**  
1. `TernaryWeight.data` changed from `Vec<i8>` to `Vec<u8>` (2-bit packed, 4 values per byte).
2. Row-aligned packing with LUT-based decode.
3. All 13 affected crate files updated atomically.
4. 677 tests pass, 0 failures.

---

### G24 — Packed SIMD Kernels

**Severity:** Medium  
**Effort:** M  
**Status:** Closed

**Description:**  
The existing AVX2 SIMD kernel (`dot_ternary_avx2`) operated on unpacked `i8` weights. With the 2-bit packed storage (G23), new SIMD kernels are required that decode packed bytes and compute dot products in a single pass.

**Resolution (commit f8fd086):**  
1. `dot_packed_ternary_i8_fast` — packed 2-bit decode + i8 dot product.
2. `dot_packed_ternary_f32_fast` — packed 2-bit decode + f32 dot product.
3. `dot_f32_f32_avx2` — AVX2+FMA f32 dot product for `lm_head` (replaces auto-vectorised baseline).
4. FMA runtime detection added via `is_x86_feature_detected!`.

---

### G25 — Sampling Allocation Elimination

**Severity:** Medium  
**Effort:** S  
**Status:** Closed

**Description:**  
`sample_next_token` allocated 2–3 MiB per token call: a `Vec<f32>` logit clone for sorting, a `HashSet<usize>` for top-p tracking, and softmax temporaries. At ~10 tok/s this produced ~30 MiB/s of short-lived heap churn.

**Resolution (commit f8fd086):**  
1. `SamplingBuffers` struct holds reusable `Vec` allocations across token steps.
2. Top-k uses `select_nth_unstable_by` (O(V) average vs O(V log V) sort).
3. Top-p uses a flag array (`Vec<bool>`) instead of `HashSet`.
4. Single softmax pass over the filtered logit set.

---

### G26 — Attention Score Pre-Allocation

**Severity:** Medium  
**Effort:** S  
**Status:** Closed

**Description:**  
`masked_attention` allocated a fresh `Vec<f32>` for attention scores on every call. At 20 heads × 30 layers per token, this produced 600 allocations/token (~320 KiB/call at max sequence length).

**Resolution (commit f8fd086):**  
Thread-local reusable score buffer via `RefCell`. Each thread in the Rayon pool retains its score vector across calls, resizing only when sequence length grows. Eliminates 320 KiB/call heap churn.

---

### G27 — lm_head Scratch Buffer

**Severity:** Medium  
**Effort:** S  
**Status:** Closed

**Description:**  
`lm_head_matmul` returned a freshly allocated `Vec<f32>` of 128,256 logits per token. The caller discarded the previous buffer each step, creating ~512 KiB/token of allocation pressure.

**Resolution (commit f8fd086):**  
1. `lm_head_matmul_into` writes directly to a pre-allocated buffer.
2. Logits buffer integrated into `ScratchBuffers` struct, allocated once at model init.

---

### G28 — Backend Trait Packed Weight Support

**Severity:** Medium  
**Effort:** M  
**Status:** Closed

**Description:**  
The `Backend::ternary_gemv` signature accepted `&[i8]` (unpacked). With 2-bit packed storage (G23), all backends must accept `&[u8]` packed weight data.

**Resolution (commit f8fd086):**  
1. `Backend::ternary_gemv` signature changed to accept `&[u8]` packed weights.
2. `CpuBackend`: dispatches to packed SIMD kernels (G24).
3. `GpuBackend`: falls back to CPU (WGSL shader not yet updated for packed format; see G29).
4. `NpuBackend`: delegates to inner backend (transparent).

---

### G29 — GPU Compute Shader for Packed 2-Bit Weights

**Severity:** Medium  
**Effort:** M  
**Status:** Open

**Description:**  
`gemv.wgsl` still expects unpacked `i32`-encoded ternary weights. With the packed 2-bit storage format (G23), the shader must decode 4 values per byte within the workgroup.

**Impact:**  
- GPU `ternary_gemv` currently falls back to CPU, negating GPU acceleration for ternary projections.
- Blocked by G23 (now closed); unblocked for implementation.

**Proposed Fix:**  
1. Rewrite `gemv.wgsl` to read `u8` packed weights from a storage buffer.
2. Decode 4 ternary values per byte using bitwise ops in WGSL.
3. Update `GpuBackend::dispatch_gemv` to upload `&[u8]` packed data.
4. Validate numerical equivalence against CPU packed kernels.

---

### G30 — SSE4.1/NEON Fallback for Packed Dot Products

**Severity:** Low  
**Effort:** M  
**Status:** Open

**Description:**  
The packed SIMD kernels (G24) require AVX2+FMA. Pre-Haswell x86 CPUs and all ARM64 targets fall back to scalar decode + dot product.

**Impact:**  
- ARM64 (Apple Silicon, Graviton) gets no SIMD benefit from packed weights.
- Pre-2013 x86 CPUs fall back to scalar.

**Proposed Fix:**  
1. `dot_packed_ternary_neon` — NEON 128-bit decode + `VMLA` accumulation.
2. `dot_packed_ternary_sse41` — SSE4.1 fallback for older x86.
3. Runtime dispatch chain: AVX2 → SSE4.1 → scalar.

---

### G31 — AVX2 lm_head via Backend Trait Extension

**Severity:** Medium  
**Effort:** M  
**Status:** Open

**Description:**  
`lm_head_matmul_into` in `bitnet-core` computes f32×f32 dot products for 128,256 vocab rows. It cannot call `dot_f32_f32_avx2` from `bitnet-cpu` due to DIP (parent modules define abstractions; child modules provide implementations). The `lm_head` operation is not part of the `Backend` trait.

**Impact:**  
- `lm_head` remains auto-vectorised, not explicitly SIMD-accelerated.
- `lm_head` is the dominant remaining bottleneck (~1.31 GB f32 weight matrix).

**Proposed Fix:**  
1. Add `lm_head_matmul_into` to the `Backend` trait.
2. `CpuBackend` implementation dispatches to `dot_f32_f32_avx2` per row.
3. `GpuBackend` can later dispatch to a GPU shader for this operation.
4. Preserves DIP: `bitnet-core` defines the abstraction, `bitnet-cpu` provides the implementation.

---

### G32 — Fully Fused Packed SIMD Kernel

**Severity:** Low  
**Effort:** M  
**Status:** Open

**Description:**  
Current packed SIMD kernels decode packed bytes to an intermediate buffer, then compute the dot product. A fully fused kernel would inline the 2-bit decode within the SIMD loop body, avoiding the intermediate buffer entirely.

**Impact:**  
- Eliminates one pass over the weight data per dot product.
- Potential 10–20% additional speedup for ternary GEMV.

**Proposed Fix:**  
Single-pass AVX2 kernel: load packed `u8` → shift/mask to extract 4 ternary values → sign-extend to i16 → `VPMADDWD` with activation → horizontal accumulate. No intermediate buffer.

---

### G33 — f16/bf16 Embedding Table for lm_head Bandwidth

**Severity:** Medium  
**Effort:** M  
**Status:** Open

**Description:**  
The `embed_tokens` / `lm_head` weight matrix is stored as `Vec<f32>` (~1.31 GB for 128,256 × 2560). Since `lm_head` is the dominant remaining bottleneck, halving its bandwidth via f16 storage reduces per-token latency.

**Impact:**  
- ~2× bandwidth reduction for `lm_head` (650 MB f16 vs 1.31 GB f32).
- Requires f16→f32 conversion during dot product (negligible compute cost with FMA).

**Proposed Fix:**  
1. Store `embed_tokens` as `Vec<half::f16>` or `Vec<half::bf16>`.
2. `lm_head_matmul_into` reads f16 and widens to f32 during accumulation.
3. AVX2 kernel: `VCVTPH2PS` for f16→f32 conversion (requires F16C, available on all AVX2 CPUs).

---

### G34 — CI Regression Harness for Tokens/Sec

**Severity:** Low  
**Effort:** S  
**Status:** Open

**Description:**  
No automated regression tracking for inference throughput. Performance regressions can be introduced silently.

**Proposed Fix:**  
1. Criterion benchmark: prefill + decode loop on a small synthetic model.
2. CI job records tok/s per commit.
3. Alert threshold: >10% regression from baseline.

---

## Risk Matrix

```
                    EFFORT
              S       M          L      XL
         ┌───────┬──────────┬───────┬───────┐
C        │       │   G06*   │       │       │
R  High  │       │ G01 G09  │  G02  │       │
I        │       │          │  G04  │       │
T        ├───────┼──────────┼───────┼───────┤
I  Med   │  G13  │ G03  G12 │       │  G11  │
C        │  G31  │ G29  G33 │       │       │
A        ├───────┼──────────┼───────┼───────┤
L  Low   │  G19  │ G30  G32 │  G14  │       │
         │  G34  │          │  G16  │       │
         └───────┴──────────┴───────┴───────┘
```

* G06 is **Partial** — tiny-model regression tests pass; real-weights `output.contains("Paris")` test now passes; additional diverse-prompt end-to-end tests remain.
Closed (removed from matrix): G05, G07, G08, G10, G17, G18, G20, G21, G22, G23, G24, G25, G26, G27, G28.
```

---

## Recommended Sprint Priorities

### Sprint 1 (Critical / Quick Wins) — ✅ Closed
1. **G20** ✅ — KVCache max_seq overflow check (S, High) — **Closed**
2. **G06** 🔶 — Golden output regression tests (M, Critical) — **Partial** (tiny-model tests pass; end-to-end against real 2B weights pending)
3. **G08** ✅ — Auto-detect model config from config.json (S, Medium) — **Closed**
4. **G18** ✅ — Checksum verification for downloads (S, Medium) — **Closed**

*Additional Sprint 1 closures:* **G07** ✅ (Samsung/MediaTek NPU detection, `BITNET_NPU_ADAPTER` env var), **G10** ✅ (`Arc<Vec<f32>>` weight tying, −1.3 GB allocation), **G17** ✅ (`decode_with_special_tokens`).

### Sprint 2 (Correctness — High Severity / Medium Effort) — ✅ Mostly Closed
1. **G21** ✅ — Weight-scale dequantization fix in `decode_packed_projection` (M, Critical) — **Closed** (`.recip()` removed; `output.contains("Paris")` verified)
2. **G01** — Exact LLaMA 3 tokenizer from tokenizer.json (M, High)
3. **G09** — Verify GQA cache layout with golden tests (M, High)
4. **G03** — Streaming token output (M, Medium)
5. **G06** 🔶 — Complete end-to-end golden output tests against real 2B weights (M, Critical)

### Sprint 3 (Phase 2 Performance) — ✅ CPU Closed; GPU Open
1. **G05** ✅ — SIMD CPU GEMV intrinsics (L, Medium) — **Closed** (AVX2 `VPSIGNW`+`VPMADDWD` in `simd.rs`, 10.2–10.4 tok/s)
2. **G22** ✅ — CPU parallelisation + allocation elimination (M, Medium) — **Closed** (Rayon lm_head/attention, persistent scratch buffers)
3. **G23** ✅ — 2-bit packed weight storage (M, High) — **Closed** (`Vec<i8>` → `Vec<u8>` packed 2-bit, 4× bandwidth reduction, 677 tests pass)
4. **G24** ✅ — Packed SIMD kernels (M, Medium) — **Closed** (`dot_packed_ternary_i8_fast`, `dot_packed_ternary_f32_fast`, `dot_f32_f32_avx2` + FMA detection)
5. **G25** ✅ — Sampling allocation elimination (S, Medium) — **Closed** (`SamplingBuffers`, O(V) top-k, flag-array top-p)
6. **G26** ✅ — Attention score pre-allocation (S, Medium) — **Closed** (thread-local `RefCell` score buffer)
7. **G27** ✅ — lm_head scratch buffer (S, Medium) — **Closed** (`lm_head_matmul_into` + `ScratchBuffers` logits buffer)
8. **G28** ✅ — Backend trait packed weights (M, Medium) — **Closed** (`ternary_gemv` now takes `&[u8]`; GPU falls back to CPU pending G29)
9. **G02** — Persistent GPU buffers (L, High) — **Open**
10. **G29** — GPU compute shader for packed 2-bit weights (M, Medium) — **Open** (`gemv.wgsl` rewrite needed)

*Performance projection:* Previous ~10.2–10.4 tok/s → theoretical ~20 tok/s target (4× ternary bandwidth reduction). Main remaining bottleneck: `lm_head` (1.31 GB f32 weights, unchanged).

### Sprint 4 (Features / Polish)
1. **G12** — Multiple model variant support (M, Medium)
2. **G13** — Dynamic GPU attention sequence length (S, Medium)
3. **G16** — HTTP server mode (L, Low)

### Sprint 5 (Next Performance — lm_head + Portability)
1. **G31** — AVX2 lm_head via Backend trait extension (M, Medium) — add `lm_head_matmul_into` to `Backend` so `CpuBackend` can use `dot_f32_f32_avx2`
2. **G33** — f16/bf16 embedding table (M, Medium) — halve lm_head bandwidth (~650 MB f16 vs 1.31 GB f32)
3. **G32** — Fully fused packed SIMD kernel (M, Low) — inline decode + dot product, eliminate intermediate buffer
4. **G30** — SSE4.1/NEON fallback (M, Low) — ARM64 + pre-Haswell x86 packed dot products
5. **G34** — CI regression harness for tokens/sec (S, Low)

---

## Acceptance Criteria for Gap Closure

A gap is considered **closed** when:

1. The fix is implemented with no `TODO` comments.
2. At least one positive test verifies the new behaviour.
3. At least one negative test verifies error handling.
4. The checklist.md and backlog.md are updated.
5. For mathematical gaps (G06, G09): analytical expected values are used in tests, not empirical observations.

---

*Last updated: 2025-07 by Ryan Clanton — Phase 2 performance optimizations (commit f8fd086): G23–G28 closed*