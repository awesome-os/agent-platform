---
title: "Model Spec 01 — Inkling 2 Mini (Looped MoE Transformer)"
slug: "model-01-inkling-2-mini"
updated: "2026-11-21"
license: "Apache-2.0"
origin: "Thinking Machines Lab (@thinkymachines)"
---

# Model Spec 01 — Inkling 2 Mini (Looped MoE Transformer)

> **Upstream:** `thinkingmachines/Inkling-2-Mini` (Thinking Machines Lab, released November 2026, **Apache 2.0**)  
> **Sibling:** `thinkingmachines/Inkling-2-Small` (`~594B` total / `~17–18B` active, 48 layers, 8/384 experts, `>=500 GB` NVFP4 / `>=800 GB` MXFP8)  
> **Runtimes:** `llama.cpp` (GGUF + `mmproj`), `EXO Labs`, `Ollama`, `vLLM`, `SGLang`, HF `transformers` (`inkling2`)

---

## 1. Verified `config.json` & Model Card Specifications

| Dimension | Verified Value | Source |
| :--- | :--- | :--- |
| **Total Physical Parameters** | **~26.4B** | Thinking Machines / HF Model Card |
| **Base Active Parameters / Pass** | **~3.4B** unique physical weights | Thinking Machines / Artificial Analysis |
| **Effective Active Parameters** | **~5.4B** (3× looped core; `~1.0B` effective per extra pass) | Thinking Machines / vLLM |
| **Physical Layers** | **24** (`0–5` Prelude, `6–17` Looped Core, `18–23` Coda) | vLLM / Model Card |
| **Logical Layer Applications** | **48** per token ($6 + 3 \times 12 + 6 = 48$) | vLLM / Model Card |
| **MoE Routing** | **8 active of 128 routed experts** per layer | Thinking Machines |
| **Hidden Size (`hidden_size`)** | **`2304`** | `config.json` (`text_config`) |
| **Attention Heads (`num_attention_heads`)** | **`24`** | `config.json` (`text_config`) |
| **KV Heads (`num_key_value_heads`)** | **`8`** (3:1 GQA) | `config.json` (`text_config`) |
| **Vocabulary Size (`vocab_size`)** | **`201024`** | `config.json` (`text_config`) |
| **Context Window** | **1,000,000 tokens** native | Model Card / Unsloth |
| **Multimodality** | Native **Text + Image** (ViT encoder → shared decoder hidden space; `mmproj`) | Model Card |
| **Native Quantization** | **NVFP4** (routed experts layers `0–19`) + **MXFP8** (layers `20–23`) | vLLM / Model Card |
| **Total Memory Footprint** | **~20 GB** (including bundled **DSpark** speculator; runs on 32 GB+ RAM) | Thinking Machines / vLLM |
| **Reasoning Control** | Tunable **`effort`** parameter (e.g., `effort 0.7`) | EXO / Runtime CLI |

---

## 2. Semantics of Operations Breakdown

1. **Looped Transformer Core (`looped_core(6, 12×3, 6, gateRank=64)`):**
   - Physical weights for layers `6–17` are executed **3 times sequentially** per token.
   - Each pass maintains its **own independent KV cache**.
   - From pass 2 onward, each sublayer reads inter-pass messages from the prior pass's attention-residual blocks modulated by a **learned rank-64 gating controller** (blending vs. forgetting prior-pass state).
   - **Memory-Bandwidth Win:** Keeps active expert weights cache-hot on unified-memory Apple Silicon (M5 Ultra / EXO) and workstation GPUs, decoupling serial reasoning depth (48 layers) from physical weight bandwidth (26.4B).

2. **Block Attention Residuals (`block_attention_residual`):**
   - Replaces fixed additive residual streams (`x = x + f(x)`) with content-dependent, depth-wise attention / pseudo-queries over earlier block outputs (inspired by Kimi Team research).
   - Prevents representation dilution across the 3 recurrent passes.

3. **Hybrid Local Sliding-Window + Global Attention (`hybrid_local_global`):**
   - Most layers run local sliding-window attention; periodic global attention layers route long-range dependencies across the 1M-token context window without $O(N^2)$ KV-cache explosion across 48 logical passes.

4. **Mixed-Precision Native Weights (`NVFP4` + `MXFP8`):**
   - Heavy routed-expert matrices in layers `0–19` use 4-bit **NVFP4**; output-sensitive coda layers `20–23` remain **MXFP8**.

5. **Bundled `DSpark` Speculative Decoder:**
   - Extends parallel-draft decoding (DFlash) with a **low-rank Markov logit-bias head** (restoring intra-block token dependencies) and a **per-position acceptance confidence head** for dynamic draft scheduling.

---

## 3. Verified Benchmarks

| Benchmark | Inkling 2 Mini |
| :--- | :---: |
| **SWE-bench Verified** | **`79.2%`** |
| **LiveCodeBench** | **`85.5%`** |
| **GPQA Diamond** | **`86.6%`** |
| **DeepSWE (v1.1)** | **`48.2%`** |

---

## 4. Explicit Unknowns (Unassailable Boundary)

The following details were **not** verified in public primary sources at archival time and are intentionally left blank rather than extrapolated:
- Exact pre-training token count, dataset mixture, and RL post-training recipe.
- Exact local-to-global attention layer ratio specifically for **Mini** (flagship 975B Inkling uses a 5:1 ratio across 66 layers; Mini's exact ratio was not published in the retrieved excerpt).
- Exact standalone AIME 2025/2026 sub-score.

---

## 5. Awesome-OS Preset Stamp Command

```powershell
.\bin\b8g-moe.ps1 stamp inkling-2-mini
```
