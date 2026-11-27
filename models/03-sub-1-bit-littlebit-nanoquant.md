---
title: "Model Spec 03 — Sub-1-Bit Quantization (LittleBit / LittleBit-2 / NanoQuant)"
slug: "model-03-sub-1-bit-littlebit-nanoquant"
updated: "2026-11-21"
origin: "Samsung Research (NeurIPS / ICML)"
---

<!-- DOC-STATIC-CORPUS -->

# Model Spec 03 — Sub-1-Bit Quantization (`LittleBit` / `LittleBit-2` / `NanoQuant`)

> **Consolidated from:** Blog #24 ([`24-sub-1-bit-llm-quantization-littlebit-and-b8g-edge-inference.md`](../awesome-os-development-history-blog/24-sub-1-bit-llm-quantization-littlebit-and-b8g-edge-inference.md)).

---

## 1. Mathematical Mechanism (`0.1–0.5 BPW`)

A single isolated weight cannot occupy 0.1 bits. Sub-1-bit compression factorizes each weight matrix $W \in \mathbb{R}^{m \times n}$ into low-rank 1-bit binary factors:

1. **Low-Rank Decomposition:** $W \approx U \cdot S \cdot V^T$ with $U \in \mathbb{R}^{m \times r}$, $V \in \mathbb{R}^{n \times r}$, rank $r \ll \min(m, n)$.
2. **1-Bit Sign Binarization:** Store $\operatorname{sign}(U), \operatorname{sign}(V) \in \{-1, +1\}$ (1 bit per factor element) plus tiny FP16 row/column/rank scaling vectors.
3. **Amortized Bit-Width:** Because $r(m + n) \ll m \cdot n$, effective storage drops to **`0.1–0.5` bits per weight** (~31× reduction vs. FP16; a 13B model fits in `< 0.9 GB` RAM).
4. **Runtime Kernel:** $W x \approx U(S(V^T x))$ reduces to two bit-packed **XNOR + popcount / LUT** inner products.

---

## 2. Upstream Variants

| Paper | Venue | Key Contribution |
| :--- | :--- | :--- |
| **LittleBit** (Banseok Lee et al.) | NeurIPS | **Dual-SVID** decomposition + multi-scale compensation + QAT down to `~0.1 BPW` |
| **LittleBit-2** (Banseok Lee, Youngmin Kim et al.) | ICML | **Joint-ITQ** orthogonal rotation to align latent geometry & suppress spiky outliers before binarization |
| **NanoQuant** (Hyochan Chong, Dongkyu Kim et al.) | ICML | **ADMM-based PTQ** for sub-1-bit / 1-bit low-rank binary factorization without full QAT retraining |

---

## 3. Honest Quantization Regime Matrix (Unassailable Standard)

| Regime | 13B Footprint | Strengths | Honest Limits | Role in Awesome-OS |
| :--- | :---: | :--- | :--- | :--- |
| **FP16 (16 BPW)** | `~26.0 GB` | Reference precision | Datacenter VRAM | Training / reference |
| **MXFP8 / NVFP4 (Mixed)** | `~7.0–13.0 GB` | Native frontier quality (Inkling 2 Mini) | Needs 16–32 GB RAM | Primary local coding & vision agent |
| **GGUF `Q4_K_M` (~4.5 BPW)** | `~7.5 GB` | Near-lossless coding & reasoning | Heavy for 4–8 GB SBCs | Standard local workstation coding |
| **BitNet 1.58-bit** | `~2.6 GB` | Strong ternary quality/size | Needs native ternary pretraining | Mid-tier local assistant |
| **LittleBit / NanoQuant (0.1–0.5 BPW)** | **`0.4–0.9 GB`** | **Fits in `< 1 GB` on \$35–\$120 CPUs/SBCs** | Degrades on deep multi-step coding/math | **Always-resident MoE router, intent classifier, TV/kiosk UI control, RAG reranker** |

---

## 4. Awesome-OS Preset Stamp Command

```powershell
.\bin\b8g-moe.ps1 stamp b8g-edge-littlebit
```
