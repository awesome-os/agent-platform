# Model Kalaschnikow (`v0.1-pre.1`)

> **Next-Generation Asymmetric Tapered Transformer atop B8G Zero-Copy Memory Arena & MicroABI Kernel**

`@awesome-os/model-kalaschnikow` is a frontier transformer architecture designed to test and deploy **asymmetric layer-wise dimension tapering** against a uniform symmetric baseline under strict **IsoFLOP compute parity** ($C \approx 6ND$).

Eliminating external Python/PyTorch dependencies, Model Kalaschnikow executes directly on the **B8G UniversalABI Microkernel** and **Zero-Copy Memory Arena (`B8G_ARENA_V1`)**, backed by `SharedArrayBuffer` and direct TypeScript toolchains via `stealifylang`.

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │ Token & Position Embedding (dim: 768)   │
                      └───────────────────┬────────────────────┘
                                          │
    ┌─────────────────────────────────────▼─────────────────────────────────────┐
    │  INGESTION LAYERS [0..3] (High Sparsity: 50%, Narrow Width: 512, 8 heads) │
    │  - Minimal bandwidth overhead for raw token representations               │
    │  - 50% Block-Sparse FLOP savings via 16x16 Frobenius energy pruning       │
    └─────────────────────────────────────┬─────────────────────────────────────┘
                                          │
    ┌─────────────────────────────────────▼─────────────────────────────────────┐
    │  REASONING CORE [4..9] (Expanded Width: 640 -> 1024, Up to 16 heads)      │
    │  - Maximum representational capacity for multi-hop semantic binding       │
    │  - Balanced 20-30% sparsity preserving core reasoning pathways            │
    └─────────────────────────────────────┬─────────────────────────────────────┘
                                          │
    ┌─────────────────────────────────────▼─────────────────────────────────────┐
    │  TOP EXTRACTION LAYERS [10..11] (Tapered Width: 768, 12 heads)            │
    │  - Focused projection onto vocabulary distribution                        │
    └─────────────────────────────────────┬─────────────────────────────────────┘
                                          │
                      ┌───────────────────▼────────────────────┐
                      │ Final LayerNorm + LM Head (vocab 32k)  │
                      └────────────────────────────────────────┘
```

---

## Key Capabilities & Pillars

### 1. Strict IsoFLOP Parity Balancing ($C \approx 6ND$)
Standard transformer designs employ uniform layer widths ($d_{\text{model}} = 768$, $d_{\text{ff}} = 3072$). When tapering width across depth, parameter counts fluctuate wildly, confounding token-efficiency gains with raw compute variance.
- Model Kalaschnikow features an **automated dimension-balancing engine** (`IsoFLOPGeometryEngine`) that dynamically computes and adjusts intermediate feedforward dimensions ($d_{\text{ff}}$).
- Guarantees that the candidate asymmetric model matches the 80M symmetric baseline within a strict **$\pm 0.70\%$ envelope** (well within the $\pm 2.0\%$ target gate).

### 2. Block-Sparse Routing & Real-Time Tile Pruning
- Partitions linear weight matrices into structured **$16 \times 16$ tiles**.
- Evaluates the Frobenius energy norm for each tile:
  $$E_b = \sqrt{\sum_{i=1}^{16}\sum_{j=1}^{16} W_{i,j}^2}$$
- Automatically prunes the lowest-energy blocks to achieve targeted layer sparsity profiles (50% in ingestion layers, 30% in reasoning layers, 20% in top layers).
- Yields immediate theoretical FLOP reductions up to **50% per layer**.

### 3. B8G Zero-Copy Memory Arena (`B8G_ARENA_V1`)
- All weight tensors, gradients, and activation scratchpads reside in a flat, contiguous `SharedArrayBuffer`.
- Enforces strict **64-byte CPU cache-line alignment** across every tensor descriptor.
- Zero V8/Node.js garbage collection overhead during token forward passes.
- Instant round-trip persistence via single-syscall binary snapshots (`.bin`) with `B8G_ARENA_V1` magic headers, ready for zero-overhead direct `mmap` into WASM and native C-ABI runtimes.

### 4. Automated Chinchilla Scaling-Law Extrapolator
- Solves parametric loss surfaces across sub-scale checkpoints:
  $$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$
- Analytically solves for optimal token allocations at the target scale:
  $$G = \frac{\beta A}{\alpha B}, \quad D_{\text{opt}} = G \cdot N^{\alpha/\beta}$$
- Predicts compute-optimal token budgets, expected cross-entropy loss, and total training FLOP budgets ($C \approx 6ND$) scaling from 80M to 7B and 70B parameter targets.

---

## Project Structure

```
repository/model-kalaschnikow-v0.1-pre.1/
├── maintenance.mts           # Single point of control runner (test, benchmark, export)
├── package.json              # Package manifest exposing clean stealifylang scripts
├── tsconfig.json             # Native TypeScript configuration (noEmit, erasableSyntaxOnly)
├── README.md                 # Complete architectural specification
├── src/
│   ├── index.mts             # Unified module exports
│   ├── types.mts             # Layer geometries, arena headers, scaling types
│   ├── arena.mts             # B8GModelArena (B8G_ARENA_V1 SharedArrayBuffer engine)
│   ├── microabi-bridge.mts   # Zero-allocation microABI linear kernel dispatcher
│   ├── block-sparse.mts      # 16x16 block-sparse linear layer & metrics tracker
│   ├── isoflop-engine.mts    # Compute budget calculator & dynamic geometry balancer
│   ├── model.mts             # KalaschnikowModel forward runner & tensor allocator
│   └── chinchilla.mts        # Non-linear power-law scaling extrapolation engine
└── test/
    └── kalaschnikow.test.mts # Full test suite (IsoFLOP parity, arena, sparsity, scaling)
```

---

## Toolchain & Command Quickstart

All operations run natively through **`stealifylang`**:

### Run Full Test Suite
```bash
stealifylang maintenance.mts test
```
Verifies:
- IsoFLOP parity within $\pm 0.70\%$ of 80M baseline.
- 16x16 block-sparse Frobenius tile pruning and 50% FLOP reduction.
- Zero-copy tensor allocation and non-allocating forward pass.
- `B8G_ARENA_V1` binary serialization and byte-exact deserialization.
- Chinchilla 7B compute-optimal ratio extrapolation.

### Run Zero-AST Project Graph Check
```bash
stealifylang check .
```
Validates the entire TypeScript module graph and import specifiers in under **10ms**.

### Run Architecture Benchmark
```bash
stealifylang maintenance.mts benchmark
```
Prints the full layer-by-layer sparsity report, memory arena usage, and 7B scaling predictions.

### Export Binary Memory Arena
```bash
stealifylang maintenance.mts export-arena
```
Dumps contiguous `Float32Array` weights into a standalone `b8g_model_kalaschnikow_v0.1_arena.bin` file ready for high-performance WASM/C-ABI runtime ingestion.

---

## Model Research & Iteration Roadmap

- [x] **v0.1-pre.1**: Clean worktree checkout from empty root commit. Full TypeScript/B8G zero-copy memory arena, IsoFLOP parity engine, 16x16 block sparsity, and Chinchilla extrapolation engine.
- [ ] **v0.1**: WASM SIMD & WebGPU compute shaders for hardware-accelerated block-sparse GEMM.
- [ ] **v0.2**: Streaming dataset ingestion via B8G RAM-backed CAS chunker for sub-scale training loops.
- [ ] **v0.3**: Scaling geometry from 80M to 1B and 7B checkpoints based on empirical Chinchilla loss curves.
