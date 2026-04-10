# bitnet.rs — BitNet b1.58 Inference Engine in Rust

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Architecture: BitNet b1.58](https://img.shields.io/badge/model-BitNet%20b1.58%202B-green.svg)](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
[![Original: microsoft/BitNet](https://img.shields.io/badge/original-microsoft%2FBitNet-0078d4?logo=github)](https://github.com/microsoft/BitNet)

> **Based on [microsoft/BitNet](https://github.com/microsoft/BitNet)** — the official C++ inference framework for 1-bit LLMs by Microsoft Research.
> This project is an independent Rust reimplementation of the BitNet b1.58 architecture and is not affiliated with or endorsed by Microsoft.

A complete, production-quality Rust reimplementation of Microsoft's **BitNet b1.58** — the first open-source, native 1-bit Large Language Model at the 2-billion parameter scale.

Supports **CPU**, **GPU** (via `wgpu`: Vulkan / Metal / DX12), and **NPU** (Windows DirectML) backends with a unified API.

---

## Table of Contents

- [Acknowledgements](#acknowledgements)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Building](#building)
- [Usage](#usage)
  - [Download Weights](#download-weights)
  - [Generate Text](#generate-text)
  - [Interactive Chat](#interactive-chat)
- [Performance](#performance)
- [Crate Structure](#crate-structure)
- [Mathematical Foundation](#mathematical-foundation)
- [Device Support](#device-support)
- [License](#license)

---

## Acknowledgements

This project is a Rust reimplementation of **Microsoft's BitNet b1.58**, built on the research and reference implementations published by Microsoft Research.

| Original work | Link |
|---|---|
| **microsoft/BitNet** — Official C++ inference engine | <https://github.com/microsoft/BitNet> |
| **BitNet b1.58 2B4T Technical Report** (Wang et al., 2025) | <https://arxiv.org/abs/2504.12285> |
| **The Era of 1-bit LLMs** (Ma et al., 2024) | <https://arxiv.org/abs/2402.17764> |
| **bitnet-b1.58-2B-4T** — Model weights on HuggingFace | <https://huggingface.co/microsoft/bitnet-b1.58-2B-4T> |

The ternary quantisation scheme (AbsMean, W2A8), the model architecture (GQA, squared-ReLU FFN, sub-layer norms), and the packed-weight format implemented here follow the specifications and reference code published in the above works. All credit for the original research and C++ implementation belongs to the authors at Microsoft Research.

---

## Architecture Overview

BitNet b1.58 is a transformer-based decoder-only LLM where **all weight matrices use ternary values {-1, 0, +1}** (1.58 bits of information per weight), with **8-bit quantised activations**. This enables:

- **0.4 GB** non-embedding memory footprint (vs 2–5 GB for equivalent f16 models)
- **29 ms** CPU decoding latency per token (vs 48–124 ms for full-precision peers)
- **~10× energy reduction** vs BF16 inference

### Model Dimensions (2B-4T)

| Parameter | Value |
|-----------|-------|
| Hidden size | 2560 |
| Layers | 30 |
| Attention heads | 20 |
| KV heads (GQA) | 5 |
| Head dimension | 128 |
| FFN dimension | 6912 |
| Vocabulary | 128,256 (LLaMA 3) |
| Context length | 4096 tokens |
| RoPE θ | 500,000 |

### Forward Pass

```
h = embed_tokens[tokens]
for layer in 0..30:
    h = h + attention(rms_norm(h))   # Pre-norm + residual
    h = h + ffn(rms_norm(h))         # Pre-norm + residual
logits = lm_head(rms_norm(h[-1]))
```

**Attention** uses Grouped Query Attention (4 query heads per KV head) with RoPE positional encoding and a post-attention sub-layer norm before the output projection.

**FFN** uses squared ReLU gating:
```
inner = ffn_sub_norm( sqrelu(gate(x)) ⊙ up(x) )
out   = down(inner)
```

---

## Quick Start

### Prerequisites

- Rust 1.75+ (`rustup update stable`)
- For GPU: Vulkan drivers (Windows/Linux) or Metal (macOS)
- For NPU: Windows 11 with DirectML-compatible driver

### Install and Build

```bash
git clone https://github.com/ryancinsight/bitnetrs
cd bitnetrs
cargo build --release
```

### Download Weights

```bash
# Downloads microsoft/bitnet-b1.58-2B-4T to ~/.cache/bitnet/
cargo run --release --bin bitnet -- download
```

Set `HF_TOKEN` for gated repositories:
```bash
HF_TOKEN=hf_your_token cargo run --release --bin bitnet -- download
```

### Run Chat

```bash
cargo run --release --bin bitnet -- chat \
  --model ~/.cache/bitnet/microsoft__bitnet-b1.58-2B-4T/model.safetensors \
  --device cpu
```

---

## Building

```bash
# Debug build (faster compilation, slower inference)
cargo build

# Release build (full optimisations, recommended for inference)
cargo build --release

# Run all tests
cargo test --workspace

# Run tests with output
cargo test --workspace -- --nocapture

# Check for errors without building
cargo check --workspace
```

### Feature Requirements

| Feature | Requirement |
|---------|-------------|
| CPU inference | Any x86-64 or ARM64 (Rayon parallelism) |
| GPU inference | Vulkan 1.1+ / Metal 2+ / DX12 |
| NPU inference | Windows 11 + DirectML driver |

---

## Usage

### Download Weights

```
bitnet download [OPTIONS]

Options:
  --repo <REPO_ID>      HuggingFace repo ID [default: microsoft/bitnet-b1.58-2B-4T]
  --revision <REV>      Git revision [default: main]
  --cache-dir <DIR>     Local cache directory [default: ~/.cache/bitnet]
  --extra-files <FILE>  Additional files to download
```

**Examples:**

```bash
# Download default 2B model
bitnet download

# Download to a custom location
bitnet download --cache-dir /data/models

# Download a specific revision
bitnet download --revision v1.0.0
```

### Generate Text

```
bitnet generate --model <PATH> --prompt <TEXT> [OPTIONS]

Required:
  -m, --model <PATH>    Path to model.safetensors
  -p, --prompt <TEXT>   Input prompt

Options:
  -d, --device <DEV>    cpu | gpu | npu [default: cpu]
      --threads <N>     CPU threads (0 = all) [default: 0]
      --gpu-id <ID>     GPU adapter index [default: 0]
  -n, --max-tokens <N>  Max new tokens [default: 256]
      --temperature <T> Sampling temperature [default: 0.7]
      --top-p <P>       Nucleus sampling threshold [default: 0.9]
      --top-k <K>       Top-k vocabulary size [default: 50]
      --repetition-penalty <A>  Repetition penalty [default: 1.1]
      --seed <N>        PRNG seed [default: 42]
```

**Examples:**

```bash
MODEL=~/.cache/bitnet/microsoft__bitnet-b1.58-2B-4T/model.safetensors

# CPU generation
bitnet generate --model $MODEL --prompt "The Eiffel Tower is located in"

# GPU generation with higher creativity
bitnet generate --model $MODEL \
  --prompt "Write a haiku about Rust programming:" \
  --device gpu \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-tokens 64

# Greedy (deterministic) decoding
bitnet generate --model $MODEL \
  --prompt "2 + 2 =" \
  --temperature 0.01 \
  --top-k 1 \
  --max-tokens 8
```

### Interactive Chat

```
bitnet chat --model <PATH> [OPTIONS]

Required:
  -m, --model <PATH>         Path to model.safetensors

Options:
  -d, --device <DEV>         cpu | gpu | npu [default: cpu]
      --threads <N>          CPU threads [default: 0]
      --gpu-id <ID>          GPU adapter index [default: 0]
      --system-prompt <TEXT> System prompt [default: "You are a helpful AI assistant."]
  -n, --max-tokens <N>       Max tokens per response [default: 512]
      --temperature <T>      Temperature [default: 0.7]
      --top-p <P>            Top-p threshold [default: 0.9]
      --top-k <K>            Top-k size [default: 50]
      --repetition-penalty   Repetition penalty [default: 1.1]
      --no-labels            Suppress "You>" / "Assistant>" labels
```

**Examples:**

```bash
MODEL=~/.cache/bitnet/microsoft__bitnet-b1.58-2B-4T/model.safetensors

# CPU chat
bitnet chat --model $MODEL

# GPU chat with custom system prompt
bitnet chat --model $MODEL \
  --device gpu \
  --system-prompt "You are a senior Rust engineer. Answer concisely with code examples."

# NPU chat (Windows; falls back to CPU if no NPU found)
bitnet chat --model $MODEL --device npu

# Scripted (no interactive labels)
echo "What is 1 + 1?" | bitnet chat --model $MODEL --no-labels
```

**Chat commands:**

| Input | Action |
|-------|--------|
| `/reset` | Clear conversation history |
| `exit` / `quit` | Exit the chat |
| `Ctrl+D` / `Ctrl+Z` | EOF — exit the chat |

---

## Performance

The following benchmarks are based on the official BitNet b1.58 technical report.

### CPU Latency (tokens/second)

| Platform | BF16 reference | BitNet b1.58 | Speedup |
|----------|---------------|--------------|---------|
| Apple M2 (ARM) | ~20 tok/s | ~34 tok/s | 1.7× |
| Intel i9 (x86) | ~8 tok/s | ~20 tok/s | 2.5× |
| 100B model on CPU | — | 5–7 tok/s | reads at human speed |

### GPU Latency (NVIDIA A100)

| Shape | W2A8 latency | BF16 latency | Speedup |
|-------|-------------|--------------|---------|
| 2560×2560 | 13 µs | 18 µs | 1.4× |
| 13824×2560 | 19 µs | 60 µs | 3.2× |
| End-to-end (2B) | 57 ms | 188 ms | 3.3× |

### Memory Footprint

| Component | Size |
|-----------|------|
| Ternary weights (non-embedding) | **~400 MB** |
| Embedding table (BF16) | **~655 MB** (down from 1.3 GB f32) |
| KV cache (4096 ctx) | ~600 MB (f32) |
| Total working set | **~1.65 GB** |

---

## Crate Structure

```
bitnetrs/
├── Cargo.toml                    # Workspace manifest
├── README.md                     # This file
├── backlog.md                    # Sprint backlog
├── checklist.md                  # Implementation checklist
└── crates/
    ├── bitnet-core/              # Core types, quantisation, backend trait
    │   └── src/
    │       ├── error.rs          # BitNetError + Result<T>
    │       ├── config.rs         # ModelConfig, GenerationConfig
    │       ├── tensor/           # Tensor<T>, DType
    │       ├── quant/            # absmean, absmax, TernaryWeight
    │       └── backend/          # Backend trait, Device enum, math ops
    │
    ├── bitnet-cpu/               # CPU backend (Rayon-parallel)
    │   └── src/
    │       ├── gemv.rs           # Ternary GEMV
    │       ├── norm.rs           # RMSNorm
    │       ├── rope.rs           # Rotary Position Embedding + cache
    │       ├── attention.rs      # Causal GQA attention
    │       └── activation.rs     # squared_relu, softmax, sqrelu_gate
    │
    ├── bitnet-gpu/               # GPU backend (wgpu: Vulkan/Metal/DX12)
    │   └── src/
    │       ├── context.rs        # wgpu device/queue creation
    │       ├── buffer.rs         # GpuBuffer upload/download
    │       ├── pipeline.rs       # Compiled WGSL compute pipelines
    │       └── shaders/          # WGSL compute shaders
    │           ├── gemv.wgsl     # Ternary GEMV
    │           ├── norm.wgsl     # RMSNorm
    │           ├── rope.wgsl     # RoPE (Q+K combined)
    │           └── attention.wgsl # Causal GQA attention
    │
    ├── bitnet-npu/               # NPU backend (Windows DirectML)
    │   └── src/
    │       ├── detect.rs         # NPU detection heuristics
    │       └── lib.rs            # NpuBackend (wraps GPU or CPU)
    │
    ├── bitnet-weights/           # Weight loading from HuggingFace
    │   └── src/
    │       ├── safetensors.rs    # BF16 safetensors parsing
    │       ├── hf_hub.rs         # HuggingFace Hub HTTP download
    │       └── loader.rs         # Weight name mapping + quantisation
    │
    ├── bitnet-tokenizer/         # LLaMA 3 BPE tokenizer
    │   └── src/lib.rs            # Tokenizer + ChatMessage + chat template
    │
    ├── bitnet-model/             # Transformer architecture + forward pass
    │   └── src/
    │       ├── lib.rs            # BitNetModel, KVCache
    │       └── device.rs         # Backend factory: Device → Arc<dyn Backend>
    │
    ├── bitnet-inference/         # High-level inference engine
    │   └── src/lib.rs            # InferenceEngine, SamplingConfig, ChatPipeline
    │
    └── bitnet-cli/               # CLI application
        └── src/main.rs           # download / generate / chat subcommands
```

### Dependency Graph

```
bitnet-cli
    └── bitnet-inference
            ├── bitnet-model
            │       ├── bitnet-cpu
            │       ├── bitnet-gpu
            │       ├── bitnet-npu ── bitnet-cpu
            │       │                └── bitnet-gpu
            │       └── bitnet-weights ── bitnet-core
            ├── bitnet-weights
            ├── bitnet-tokenizer
            └── bitnet-core
```

---

## Mathematical Foundation

### Ternary Weight Quantisation (AbsMean)

Given a full-precision weight matrix **W** ∈ ℝ^{M×K}:

```
α_W  = mean(|W|)                           # per-tensor scale
W_q  = clip( round( W / α_W ), −1, 1 )    # ternary {-1, 0, +1}
```

Effective weight: `W_eff[i,j] = W_q[i,j] × α_W`

### 8-bit Activation Quantisation (AbsMax, per-token)

```
α_x  = max(|x|) / 127                     # per-token scale
x_q  = clip( round( x / α_x ), −128, 127 )
```

### Quantised Forward Pass

```
y = (W_q @ x_q) × α_W × α_x / 127
```

The integer dot product `W_q @ x_q` uses only {-1, 0, +1} × {-128..127} — no floating-point multiply in the inner loop.

### Rotary Position Embedding (RoPE)

For head vector **x** at position `pos`, dimension pair `(2i, 2i+1)`:

```
θ_i       = pos / rope_theta^(2i / head_dim)
x'[2i]    = x[2i]   × cos(θ_i) − x[2i+1] × sin(θ_i)
x'[2i+1]  = x[2i]   × sin(θ_i) + x[2i+1] × cos(θ_i)
```

BitNet 2B uses `rope_theta = 500,000` for extended context.

### Token Sampling

Given logits **z** ∈ ℝ^V:

1. Repetition penalty: `z[t] /= α` for each `t` in context
2. Temperature: `z[t] /= τ`
3. Top-k: zero out all but top-k logits
4. Top-p: keep smallest prefix with cumulative probability ≥ p
5. Softmax → categorical sample

---

## Device Support

### CPU (`--device cpu`)

- Uses **Rayon** for data-parallel GEMV (outer loop over output rows).
- Pre-computed **RoPE cosine/sine tables** avoid redundant trigonometry.
- Numerically-stable softmax with max-subtraction.
- Configure threads: `--threads N` (default: all logical cores).

### GPU (`--device gpu`)

- Uses **wgpu** with WGSL compute shaders.
- Cross-platform: Vulkan (Windows/Linux), Metal (macOS), DX12 (Windows).
- Each shader uses workgroup-level shared memory for parallel reductions.
- Falls back to CPU for operations without a GPU shader implementation.
- Select adapter: `--gpu-id N` (default: 0 = highest-performance GPU).

### NPU (`--device npu`)

- On **Windows**: probes for DirectML-backed wgpu adapters (Intel, AMD, Qualcomm NPUs).
- Detection uses adapter name heuristics (keywords: `"npu"`, `"neural"`, `"ai boost"`, etc.).
- Transparent CPU fallback if no NPU is detected — no error is returned.
- Adapter index: `--gpu-id` is ignored; uses detected NPU at index 0.

### Environment Variables

| Variable | Effect |
|----------|--------|
| `RUST_LOG` | Log level (e.g. `info`, `debug`, `bitnet=trace`) |
| `HF_TOKEN` | HuggingFace access token for gated repos |
| `BITNET_CACHE` | Override default cache directory |

---

## License

MIT License — see [LICENSE](LICENSE).

---

## References

- Wang, S. et al. **BitNet b1.58 2B4T Technical Report** (April 2025). <https://arxiv.org/abs/2504.12285>
- Ma, S. et al. **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits** (February 2024). <https://arxiv.org/abs/2402.17764>
- **microsoft/BitNet** — Official C++ inference framework for 1-bit LLMs. <https://github.com/microsoft/BitNet>
- **microsoft/bitnet-b1.58-2B-4T** — Pre-trained model weights on HuggingFace. <https://huggingface.co/microsoft/bitnet-b1.58-2B-4T>
- **LLaMA 3** — Tokenizer architecture (Meta AI). <https://ai.meta.com/blog/meta-llama-3/>

---

*Built with ❤️ in Rust by [Ryan Clanton](https://github.com/ryancinsight)*