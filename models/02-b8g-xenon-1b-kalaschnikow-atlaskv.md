---
title: "Model Spec 02 — B8G Xenon 1B / Kalaschnikow + AtlasKV"
slug: "model-02-b8g-xenon-1b"
updated: "2026-11-21"
license: "Awesome-OS Sovereign"
origin: "Awesome-OS Core (repository/model-kalaschnikow-v0.1-pre.1 + sections/atlas-kv)"
---

<!-- DOC-STATIC-CORPUS -->

# Model Spec 02 — B8G Xenon 1B / Kalaschnikow + AtlasKV

> **Consolidated from:** Blog #15 ([`15-b8g-xenon-1b-cognitive-core-and-atlaskv-memory.md`](../awesome-os-development-history-blog/15-b8g-xenon-1b-cognitive-core-and-atlaskv-memory.md)), [`repository/model-kalaschnikow-v0.1-pre.1/`](../../repository/model-kalaschnikow-v0.1-pre.1/src/model.mts), and [`sections/atlas-kv/`](../../sections/atlas-kv/xenon-1b-inference.host.mts).

---

## 1. Architectural Thesis

Stop wasting 90% of model weights memorizing static facts. Split intelligence cleanly into:
1. **A Compact Tapered Cognitive Core (`KalaschnikowModel`)** running in 64B-aligned `B8G_ARENA_V1` linear memory (`Softmax`, `GEMM`, `Block-Sparse MLP`, `SLERP`).
2. **Externalized Parametric Memory (`sections/atlas-kv`)** via `KG2KV` + `HiKVP` hierarchical cluster pruning backed by `.b8p` SHA-256 CAS and guarded by per-cluster `CapabilityToken<TID>` tenant isolation at attention time.

---

## 2. Semantics of Operations Breakdown

| Axis | Specification |
| :--- | :--- |
| **Topology** | `asymmetric_tapered` (layer-wise tapered `dModelIn -> dModelOut`, e.g. `768 -> 256 -> 128` in compact benchmark profile; scales to 1B cognitive core) |
| **Residual** | `additive` with learned projection bridges across tapered dimensions |
| **Attention** | Multi-head attention over 64B-aligned `B8GModelArena` buffers (`B8G_ARENA_V1`) |
| **FFN** | `block_sparse` (`sparsity = 0.5`, `blockSize = 16`) |
| **Quantization** | `float32` reference / `Q4_K_M` or `littlebit_sub1` deployment target |
| **Parametric Memory** | `atlaskv_hikvp` (`128` clusters, `top-2` pruning, `CapabilityToken<TID>` isolation) |
| **Speculator** | `none` (ultra-low latency core) |
| **Boundary Gate** | `CapabilityToken<TID>` tenant scope gate on memory clusters |

---

## 3. Verified `sections/atlas-kv` Benchmark (`50,000` Triples, `128` Clusters)

| Metric | $O(N)$ Flat Scan | `B8G Xenon 1B` HiKVP (`top-2`) | Gain |
| :--- | :---: | :---: | :---: |
| **40-Query Latency** | `68.55 ms` | **`2.73 ms`** | **`25.1× faster`** |
| **Active Working-Set Memory** | `6,250 KB` | **`114 KB`** | **`-98.2%` VRAM** |
| **Top-1 Recall** | `100%` | **`100%` (`1.0`)** | Zero accuracy drop |
| **Tenant Isolation** | None | **`CapabilityToken<TID>`** | Private clusters unreachable |

---

## 4. Awesome-OS Preset Stamp Command

```powershell
.\bin\b8g-moe.ps1 stamp b8g-xenon-1b
```
