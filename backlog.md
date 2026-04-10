# bitnet.rs — Sprint Backlog

## Vision

Complete, production-quality Rust reimplementation of Microsoft's BitNet b1.58 (2B-4T) with CPU, GPU (wgpu), and NPU (DirectML) support. Every line mathematically justified, architecturally sound, and fully tested.

---

## Epics

### E1 — Core Quantisation Mathematics
**Status:** ✅ Complete

| ID | Story | Priority | Status |
|----|-------|----------|--------|
| E1-1 | `TernaryWeight` struct with validated construction and bit-packing | P0 | ✅ Done |
| E1-2 | `absmean_quantize` (f32/bf16 → ternary i8 + scale) | P0 | ✅ Done |
| E1-3 | `absmax_quantize_row` (f32 → i8 + scale, per-token) | P0 | ✅ Done |
| E1-4 | `pack_ternary` / `unpack_ternary` (2-bit packing) | P1 | ✅ Done |
| E1-5 | `ternary_dot_product_quantised` end-to-end pipeline | P1 | ✅ Done |
| E1-6 | Property tests: roundtrip, error bounds, ternary invariants | P0 | ✅ Done |

---

### E2 — Backend Abstraction Layer
**Status:** ✅ Complete

| ID | Story | Priority | Status |
|----|-------|----------|--------|
| E2-1 | `Backend` trait: ternary_gemv, rms_norm, rope_embed, masked_attention, squared_relu, softmax, elementwise_mul | P0 | ✅ Done |
| E2-2 | `Device` enum: Cpu/Gpu/Npu with convenience constructors | P0 | ✅ Done |
| E2-3 | Blanket `impl Backend for Arc<dyn Backend>` | P1 | ✅ Done |
| E2-4 | Standalone math ops: rms_norm_f32, rope_cos_sin_table, softmax_f32, squared_relu_f32, lm_head_matmul | P0 | ✅ Done |

---

### E3 — CPU Backend
**Status:** ✅ Complete

| ID | Story | Priority | Status |
|----|-------|----------|--------|
| E3-1 | `CpuBackend` struct with Rayon thread pool initialisation | P0 | ✅ Done |
| E3-2 | `ternary_gemv_f32` — Rayon-parallel outer loop over rows | P0 | ✅ Done |
| E3-3 | `ternary_gemv_quantised` — integer accumulator W·x variant | P1 | ✅ Done |
| E3-4 | `rms_norm` — RMSNorm with ε validation | P0 | ✅ Done |
| E3-5 | `rms_norm_inplace` — in-place variant | P1 | ✅ Done |
| E3-6 | `apply_rope` — RoPE at single sequence position | P0 | ✅ Done |
| E3-7 | `RopeCache` — pre-computed cos/sin table for all positions | P1 | ✅ Done |
| E3-8 | `apply_rope_cached` — fast path using cached tables | P1 | ✅ Done |
| E3-9 | `masked_attention` — causal GQA scaled dot-product attention | P0 | ✅ Done |
| E3-10 | `squared_relu`, `softmax`, `sqrelu_gate` activations | P0 | ✅ Done |
| E3-11 | `impl Backend for CpuBackend` | P0 | ✅ Done |
| E3-12 | Full mini-transformer-block smoke test | P0 | ✅ Done |

---

### E4 — GPU Backend (wgpu)
**Status:** ✅ Complete

| ID | Story | Priority | Status |
|----|-------|----------|--------|
| E4-1 | `create_wgpu_device` — adapter enumeration and device creation | P0 | ✅ Done |
| E4-2 | `GpuBuffer` — typed upload/download wrappers | P0 | ✅ Done |
| E4-3 | `gemv.wgsl` — ternary GEMV compute shader (workgroup reduction) | P0 | ✅ Done |
| E4-4 | `norm.wgsl` — RMSNorm compute shader (2-phase reduction) | P0 | ✅ Done |
| E4-5 | `rope.wgsl` — RoPE compute shader (Q+K combined entry point) | P0 | ✅ Done |
| E4-6 | `attention.wgsl` — causal GQA attention shader (online softmax) | P0 | ✅ Done |
| E4-7 | `GpuPipelines` — compile all WGSL shaders at init time | P0 | ✅ Done |
| E4-8 | `GpuBackend` struct + `impl Backend` with CPU fallback | P0 | ✅ Done |
| E4-9 | GPU dispatch helpers: dispatch_gemv, dispatch_rms_norm, dispatch_rope, dispatch_attention | P0 | ✅ Done |
| E4-10 | Shader source content tests (entry points, bindings present) | P1 | ✅ Done |

---

### E5 — NPU Backend (Windows DirectML)
**Status:** ✅ Complete

| ID | Story | Priority | Status |
|----|-------|----------|--------|
| E5-1 | `detect_npu` — wgpu adapter enumeration with name heuristics | P0 | ✅ Done |
| E5-2 | `detect_all_npus` — return all NPU candidates sorted by priority | P1 | ✅ Done |
| E5-3 | `NpuInfo` struct with vendor/type/adapter_index | P0 | ✅ Done |
| E5-4 | `NpuVendor`, `NpuAdapterType` enums | P1 | ✅ Done |
| E5-5 | `NpuBackend` — wraps GPU (DirectML) or CPU fallback | P0 | ✅ Done |
| E5-6 | `impl Backend for NpuBackend` — transparent delegation | P0 | ✅ Done |
| E5-7 | Cross-backend consistency test (NPU output == CPU output when no NPU) | P1 | ✅ Done |

---

### E6 — Weight Loading
**Status:** ✅ Complete

| ID | Story | Priority | Status |
|----|-------|----------|--------|
| E6-1 | `parse_safetensors_header` — JSON header parsing | P0 | ✅ Done |
| E6-2 | `load_bf16_safetensors` — full file → HashMap<String, Vec<f32>> | P0 | ✅ Done |
| E6-3 | BF16/F16/F32 byte → f32 conversion with non-finite replacement | P0 | ✅ Done |
| E6-4 | `load_raw_safetensors` — raw bytes + shape + dtype | P1 | ✅ Done |
| E6-5 | `load_safetensors_meta` — header-only metadata (fast path) | P1 | ✅ Done |
| E6-6 | `LayerWeights` — per-layer weight struct | P0 | ✅ Done |
| E6-7 | `ModelWeights` — full model weight container | P0 | ✅ Done |
| E6-8 | `load_weights_from_bf16` — HF tensor name → model struct mapping | P0 | ✅ Done |
| E6-9 | `quantise_weight` — absmean quantisation on load | P0 | ✅ Done |
| E6-10 | `download_model_from_hf` — async HTTP download with progress bars | P0 | ✅ Done |
| E6-11 | Cache hit detection and retry logic | P1 | ✅ Done |
| E6-12 | Blocking wrapper `download_model_from_hf_blocking` | P1 | ✅ Done |

---

### E7 — Tokenizer
**Status:** ✅ Complete

| ID | Story | Priority | Status |
|----|-------|----------|--------|
| E7-1 | `Tokenizer::llama3` — cl100k_base BPE initialisation | P0 | ✅ Done |
| E7-2 | `encode` — text → Vec<u32> with optional BOS | P0 | ✅ Done |
| E7-3 | `decode` — Vec<u32> → String (skips special tokens) | P0 | ✅ Done |
| E7-4 | `decode_single` — single token → bytes (streaming) | P1 | ✅ Done |
| E7-5 | `apply_chat_template` — LLaMA 3 Instruct format | P0 | ✅ Done |
| E7-6 | `encode_chat` — template + encode combined | P1 | ✅ Done |
| E7-7 | Special token IDs: BOS=128000, EOS=128009 | P0 | ✅ Done |
| E7-8 | Roundtrip encode/decode property tests | P0 | ✅ Done |

---

### E8 — Model Architecture
**Status:** ✅ Complete

| ID | Story | Priority | Status |
|----|-------|----------|--------|
| E8-1 | `KVCache` — per-layer K/V storage with fill/advance/clear | P0 | ✅ Done |
| E8-2 | `KVCache::store_kv` — append or overwrite at position | P0 | ✅ Done |
| E8-3 | `KVCache::k_slice` / `v_slice` — position-bounded view | P0 | ✅ Done |
| E8-4 | `BitNetModel::new` — weight + backend initialisation | P0 | ✅ Done |
| E8-5 | `BitNetModel::forward` — full 30-layer forward pass | P0 | ✅ Done |
| E8-6 | Token embedding lookup | P0 | ✅ Done |
| E8-7 | Pre-attention RMSNorm + Q/K/V projection + RoPE | P0 | ✅ Done |
| E8-8 | KV cache store + causal attention + attn_sub_norm + o_proj | P0 | ✅ Done |
| E8-9 | Pre-FFN RMSNorm + gate/up proj + sqrelu gate | P0 | ✅ Done |
| E8-10 | FFN sub-norm + down proj | P0 | ✅ Done |
| E8-11 | Final norm + lm_head_matmul (weight-tied) | P0 | ✅ Done |
| E8-12 | `create_backend` factory in `device.rs` | P0 | ✅ Done |
| E8-13 | Autoregressive decode smoke test | P0 | ✅ Done |

---

### E9 — Inference Engine
**Status:** ✅ Complete

| ID | Story | Priority | Status |
|----|-------|----------|--------|
| E9-1 | `SamplingConfig` — temperature, top_k, top_p, repetition_penalty, seed | P0 | ✅ Done |
| E9-2 | `sample_next_token` — full sampling pipeline | P0 | ✅ Done |
| E9-3 | Greedy decoding shortcut (top_k=1) | P1 | ✅ Done |
| E9-4 | `InferenceEngine::new` — weights + model + tokenizer + kv_cache | P0 | ✅ Done |
| E9-5 | `InferenceEngine::generate` — prefill + decode loop | P0 | ✅ Done |
| E9-6 | `InferenceEngine::generate_chat` — chat template + generate | P0 | ✅ Done |
| E9-7 | `InferenceEngine::reset` — KV cache clear | P1 | ✅ Done |
| E9-8 | `ChatPipeline` — stateful multi-turn conversation | P0 | ✅ Done |
| E9-9 | `ChatPipeline::chat` — add to history + generate | P0 | ✅ Done |
| E9-10 | `ChatPipeline::reset_conversation` | P1 | ✅ Done |
| E9-11 | Sampling determinism tests (same seed → same token) | P0 | ✅ Done |
| E9-12 | Sampling stress test (100 steps, no panic) | P0 | ✅ Done |

---

### E10 — CLI Application
**Status:** ✅ Complete

| ID | Story | Priority | Status |
|----|-------|----------|--------|
| E10-1 | `bitnet download` — async HF Hub download with progress | P0 | ✅ Done |
| E10-2 | `bitnet generate` — non-interactive text generation | P0 | ✅ Done |
| E10-3 | `bitnet chat` — interactive multi-turn chat REPL | P0 | ✅ Done |
| E10-4 | Device selection: cpu/gpu/npu via `--device` | P0 | ✅ Done |
| E10-5 | Sampling params: temperature, top_p, top_k, repetition_penalty, seed | P0 | ✅ Done |
| E10-6 | Cache directory resolution (arg > BITNET_CACHE env > ~/.cache/bitnet) | P1 | ✅ Done |
| E10-7 | `/reset` chat command + exit/quit handling | P1 | ✅ Done |
| E10-8 | Per-turn timing report (tok/s) | P2 | ✅ Done |
| E10-9 | `--no-labels` flag for scripting | P2 | ✅ Done |
| E10-10 | CLI argument parsing tests (clap try_parse_from) | P1 | ✅ Done |

---

### E11 — Documentation & Artifacts
**Status:** ✅ Complete

| ID | Story | Priority | Status |
|----|-------|----------|--------|
| E11-1 | `README.md` — architecture, quick start, usage, performance, crate map | P0 | ✅ Done |
| E11-2 | `backlog.md` — this file | P1 | ✅ Done |
| E11-3 | `checklist.md` — implementation status tracking | P1 | ✅ Done |
| E11-4 | `gap_audit.md` — known gaps and future work | P1 | ✅ Done |
| E11-5 | In-code Rustdoc with mathematical invariants on every public item | P0 | ✅ Done |
| E11-6 | Algorithm derivations and theorem citations in comments | P1 | ✅ Done |

---

## Backlog — Future Work (Not in Current Sprint)

### F1 — Performance Optimisations

| ID | Story | Priority | Status |
|----|-------|----------|--------|
| F1-0 | LM head `Arc<Vec<f32>>` weight tying — eliminates ~1.3 GB embedding duplication (G10) | P0 | ✅ Done |
| F1-1 | SIMD acceleration for CPU GEMV (AVX2 / NEON) | P1 | ✅ Done |
| F1-2 | Persistent GPU buffers (avoid re-upload per forward call) | P0 | — |
| F1-3 | Fused Q+K+V projection (single GEMV for wqkv) | P1 | — |
| F1-4 | Speculative decoding (draft model acceleration) | P2 | — |
| F1-5 | Flash Attention variant for GPU (O(N) memory) | P1 | — |
| F1-6 | int4 / int8 packing for smaller KV cache footprint | P2 | — |
| F1-7 | Batched inference (multiple prompts simultaneously) | P1 | — |
| F1-8 | Quantised embedding (U8) to reduce embedding table memory | P2 | — |
| F1-9 | Rayon work-stealing tuning for heterogeneous core counts | P2 | — |
| F1-13 | Rayon-parallel `lm_head_matmul` (128K vocab rows via `par_chunks`) | P1 | ✅ Done |
| F1-14 | Rayon-parallel attention heads (20 heads via `par_chunks_mut`) | P1 | ✅ Done |
| F1-15 | Persistent model scratch buffers (`ScratchBuffers` struct) | P1 | ✅ Done |
| F1-16 | O(1) incremental token context in decode loop | P1 | ✅ Done |
| F1-17 | Pre-allocated activation quantisation buffer (`absmax_quantize_row_into`) | P1 | ✅ Done |
| F1-10 | KV cache overflow enforcement against `max_seq` | P0 | ✅ Done |
| F1-11 | NPU detection keyword expansion: Samsung/MediaTek/APU + `BITNET_NPU_ADAPTER` env var override (G07) | P1 | ✅ Done |
| F1-12 | Activation quantisation (absmax i8) in forward-pass GEMV — required for quality output | P0 | — |

### Future Performance — Remaining Opportunities

| ID | Story | Priority | Status |
|----|-------|----------|--------|
| FP-1 | 2-bit packed weight storage (4× less memory bandwidth, ~2× speedup potential) | P1 | — |
| FP-2 | SSE4.1 fallback SIMD path for pre-AVX2 CPUs | P2 | — |
| FP-3 | GPU inference via wgpu compute shaders (persistent buffers, see G02) | P0 | — |
| FP-4 | KV cache quantisation (fp16 or int8) | P2 | — |

---

### F2 — Additional Model Variants

| ID | Story | Priority | Status |
|----|-------|----------|--------|
| F2-1 | BitNet b1.58 0.7B (bitnet_b1_58-large) | P1 | — |
| F2-2 | BitNet b1.58 3.3B (bitnet_b1_58-3B) | P1 | — |
| F2-3 | Llama3-8B-1.58-100B-tokens | P2 | — |
| F2-4 | Falcon3 family (1B–10B, tiiuae) | P2 | — |
| F2-5 | Dynamic model config detection from config.json | P0 | ✅ Done |

### F3 — Features

| ID | Story | Priority | Status |
|----|-------|----------|--------|
| F3-1 | Streaming token output (tokio channel per token) | P1 | — |
| F3-2 | Server mode: HTTP API (OpenAI-compatible /v1/chat/completions) | P1 | — |
| F3-3 | Beam search sampling | P2 | — |
| F3-4 | Constrained generation (grammar / JSON schema) | P2 | — |
| F3-5 | Context window extension (RoPE NTK scaling) | P2 | — |
| F3-6 | Model quantisation export (GGUF format) | P2 | — |
| F3-7 | Benchmark harness (tokens/sec, energy, perplexity) | P1 | — |
| F3-8 | System info CLI (`bitnet info` — show adapters, memory) | P2 | — |
| F3-9 | `Tokenizer::decode_with_special_tokens` — special token string rendering for debugging (G17) | P1 | ✅ Done |
| F3-10 | HF Hub SHA256 sidecar checksum verification — detect corrupted/tampered downloads (G18) | P0 | ✅ Done |

### F4 — Testing

| ID | Story | Priority | Status |
|----|-------|----------|--------|
| F4-1 | Property tests with proptest for all quantisation functions | P1 | — |
| F4-2 | Golden output tests: compare against reference Python implementation | P0 | 🔶 Partial |
| F4-3 | End-to-end integration test: download + generate deterministic output | P0 | — |
| F4-4 | GPU vs CPU numerical equivalence tests (tolerances) | P1 | — |
| F4-5 | Criterion benchmarks for GEMV, attention, norm | P1 | — |
| F4-6 | Adversarial inputs: max-length context, all-same tokens, OOV handling | P2 | — |

---

## Priority Legend

| Symbol | Meaning |
|--------|---------|
| P0 | Must-have for correctness/functionality |
| P1 | Should-have for production quality |
| P2 | Nice-to-have / future improvement |

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ Done | Implemented and tested |
| 🚧 In Progress | Currently being implemented |
| ⏳ Planned | Scheduled for current sprint |
| ❌ Blocked | Blocked by dependency |
| — | Not started |