---
title: "Awesome-OS Unified Model Architecture & Semantics of Operations Atlas"
slug: "b8g-models-atlas"
updated: "2026-11-21"
motto: "Nunquam ante, numquam iterum"
---

# Awesome-OS Unified Model Architecture Atlas (`docs/models/`)

> **Single Source of Truth for Model Implementation Details, Semantics of Operations, Research Model Stamping, and Boundary-Verified MoE Execution.**

Previously, model architecture notes were scattered across blog posts (`15-b8g-xenon-1b`, `24-sub-1-bit-llm-quantization`), `repository/model-kalaschnikow-v0.1-pre.1/`, `sections/atlas-kv/`, `intersections/b8g-brain/`, and `docs/research/`. This directory unifies **all** model implementation details under one canonical taxonomy so we can:

1. **Curate** verified external & internal model architectures strictly under the **Unassailable vs. Assailable Standard** (exact numbers from `config.json`/papers, explicit unknowns, zero hype).
2. **Decompose** every model into composable **Semantics of Operations** (`sections/b8g-model-ops/`) — Topology/Looping, Residual Algebra, Attention Kernel, FFN/MoE Routing, Quantization, Parametric Memory, Speculative Decoding, and Boundary Gates.
3. **Stamp Out** new research models programmatically (`.\bin\b8g-moe stamp <preset>`) with deterministic parameter counts, FLOPs/token, and `B8G_ARENA_V1` memory layouts.
4. **Execute & Verify** the **First Real Boundary-Verified MoE** (`intersections/b8g-moe-gate/`), where every expert is a `B8G-SECT-1` Section guarded by hard ingress/egress mathematical & capability gates.

---

## 1. Curated Model Specification Index

| ID | Model Spec | Family / Origin | Params (Total / Active) | Key Semantic Innovations | Target Runtime / Code |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **01** | [Inkling 2 Mini](./01-inkling-2-mini-looped-moe.md) | Thinking Machines Lab (Apache 2.0) | `26.4B` / `3.4B` base (`5.4B` eff.) | **3× Looped Core** (24→48 logical layers, rank-64 gate), **Block Attention Residuals**, 8/128 MoE, Hybrid Local+Global 1M ctx, NVFP4+MXFP8, **DSpark** speculator | `llama.cpp` GGUF + `mmproj`, EXO, `b8g-brain` |
| **02** | [B8G Xenon 1B / Kalaschnikow + AtlasKV](./02-b8g-xenon-1b-kalaschnikow-atlaskv.md) | Awesome-OS Sovereign Core | `~1B` tapered / sparse | **Asymmetric Tapered Layers**, 64B-aligned `B8G_ARENA_V1` block-sparse GEMM/SLERP, **AtlasKV HiKVP** (`25.1×` faster, `-98.2%` VRAM, `100%` recall) + `CapabilityToken<TID>` | `repository/model-kalaschnikow-v0.1-pre.1/`, `sections/atlas-kv/` |
| **03** | [LittleBit / LittleBit-2 / NanoQuant](./03-sub-1-bit-littlebit-nanoquant.md) | Samsung Research (NeurIPS / ICML) | `0.1–0.5 BPW` (`13B` in `<0.9 GB`) | **Dual-SVID** low-rank binary factorization ($W \approx U S V^T$, 1-bit signs), Joint-ITQ outlier rotation, ADMM PTQ | Always-resident edge routers & TV/kiosk SBCs |
| **04** | [B8G Sovereign Boundary-Verified MoE](./04-boundary-verified-moe-blueprint.md) | Awesome-OS Research Blueprint | Composable (`stamp`-able) | **Section-per-Expert MoE** with hard **Ingress/Egress Boundary Gates** (capability token, finite/energy bounds, Block Attention Residuals, symbolic safe-gate verification) | `sections/b8g-model-ops/`, `intersections/b8g-moe-gate/` (`.\bin\b8g-moe`) |

---

## 2. Canonical Semantics of Operations Taxonomy

Every model in this atlas — whether curated from upstream research or stamped out for our own experiments — is expressed as an exact algebraic tuple over 7 orthogonal operation axes:

```text
ModelSpec = (
  TopologyOp,      // standard_stack | looped_core(prelude, core×K, coda, gateRank) | asymmetric_tapered
  ResidualOp,      // additive | block_attention_residual(blockSize)
  AttentionOp,     // gqa(nQ, nKV) | hybrid_local_global(window, ratio) | mla
  FfnOp,           // dense | moe_routed(activeK, totalE, sharedE) | block_sparse(sparsity, blockSize)
  QuantOp,         // fp16 | mxfp8 | nvfp4 | q4_k_m | bitnet_158 | littlebit_sub1 | mixed(layerRanges)
  MemoryOp,        // kv_cache_only | per_loop_kv_cache | atlaskv_hikvp(clusters, capToken)
  SpeculatorOp,    // none | dspark(markovRank, confidenceHead)
  BoundaryGateOp   // none | verified_moe_gate(capScope, maxL2Norm, finiteCheck, symbolicGate)
)
```

---

## 3. Executable Tooling & Cross-References

- **Model Ops Section (IR + Stamper + C11 ABI):** [`sections/b8g-model-ops/`](../../sections/b8g-model-ops/index.mts) · [`b8g_model_ops_abi.h`](../../sections/b8g-model-ops/b8g_model_ops_abi.h)
- **Boundary-Verified MoE Intersection & CLI:** [`intersections/b8g-moe-gate/`](../../intersections/b8g-moe-gate/index.mts) · `.\bin\b8g-moe.ps1 catalog | stamp <preset> | verify`
- **B8G-Brain Profile & Router:** [`intersections/b8g-brain/BRAIN-PROFILE-SCHEMA.md`](../../intersections/b8g-brain/BRAIN-PROFILE-SCHEMA.md)
- **Inference Manager (`Execution <-> Projection`):** [`intersections/inference-manager/README.md`](../../intersections/inference-manager/README.md)
- **Research Dossiers Index:** [`docs/research/README.md`](../research/README.md)
