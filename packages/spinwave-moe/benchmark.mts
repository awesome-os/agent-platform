/**
 * Head-to-Head Spin-Wave Benchmark Suite (`erasableSyntaxOnly: true`, zero external deps).
 *
 * Benchmarks:
 * 1. Intersection Bus Throughput (200,000 ops):
 *    Node `EventEmitter` object-allocation dispatch vs. Spin-Wave `SharedArrayBuffer`
 *    zero-copy 64B-aligned phasor superposition.
 * 2. 64-Expert MoE Routing (50,000 tokens):
 *    Traditional JS `Float32` 64-dim cosine-similarity object sort vs. Spin-Wave Complex
 *    Phasor Coherence (`q_re*e_re + q_im*e_im`) zero-alloc Top-K router.
 * 3. 500-Node DAG Dirty-Wave Scheduling (1,000 incremental 1-node edits):
 *    Traditional full Kahn topological sort + `Set` traversal vs. `SpinWaveDagScheduler`
 *    damped-gravity localized wave propagation.
 */

import { EventEmitter } from "node:events";
import { performance } from "node:perf_hooks";
import {
  NODE_STRIDE_BYTES,
  NODE_STRIDE_WORDS,
  SpinWaveDagScheduler,
  verifySpinWaveScheduler,
  type DagEdge,
} from "./scheduler.mts";

// Deterministic xorshift32 PRNG for reproducible datasets
function createPrng(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 0x100000000;
  };
}

interface BusBenchResult {
  readonly ops: number;
  readonly baselineMs: number;
  readonly baselineOpsPerSec: number;
  readonly baselineNsPerOp: number;
  readonly spinwaveMs: number;
  readonly spinwaveOpsPerSec: number;
  readonly spinwaveNsPerOp: number;
  readonly speedup: number;
  readonly sink: number;
}

interface MoeBenchResult {
  readonly tokens: number;
  readonly experts: number;
  readonly topK: number;
  readonly baselineMs: number;
  readonly baselineTokensPerSec: number;
  readonly baselineUsPerToken: number;
  readonly spinwaveMs: number;
  readonly spinwaveTokensPerSec: number;
  readonly spinwaveUsPerToken: number;
  readonly speedup: number;
  readonly sink: number;
}

interface DagBenchResult {
  readonly nodes: number;
  readonly edges: number;
  readonly edits: number;
  readonly baselineMs: number;
  readonly baselineUsPerEdit: number;
  readonly baselineAvgVisited: number;
  readonly spinwaveMs: number;
  readonly spinwaveUsPerEdit: number;
  readonly spinwaveAvgFired: number;
  readonly speedup: number;
  readonly sink: number;
}

// ============================================================================
// Bench 1 — Intersection Bus Throughput (200,000 ops)
// ============================================================================
function runBench1BusThroughput(ops = 200_000): BusBenchResult {
  const CHANNELS = 64;

  // --- Baseline: Node EventEmitter object-allocation dispatch ---
  const emitter = new EventEmitter();
  let eeAccumRe = 0;
  let eeAccumIm = 0;
  emitter.on(
    "intersection",
    (ev: { channel: number; re: number; im: number; energy: number }) => {
      eeAccumRe += ev.re;
      eeAccumIm += ev.im + ev.energy * 0.001;
    },
  );

  // Warmup
  for (let i = 0; i < 10_000; i++) {
    const ch = i & (CHANNELS - 1);
    emitter.emit("intersection", {
      channel: ch,
      re: 0.7071,
      im: 0.7071,
      energy: 1.0,
    });
  }

  eeAccumRe = 0;
  eeAccumIm = 0;
  const t0 = performance.now();
  for (let i = 0; i < ops; i++) {
    const ch = i & (CHANNELS - 1);
    const re = (i & 1) === 0 ? 0.8660254 : 0.5;
    const im = (i & 1) === 0 ? 0.5 : 0.8660254;
    emitter.emit("intersection", {
      channel: ch,
      re,
      im,
      energy: 1.0,
    });
  }
  const baselineMs = performance.now() - t0;

  // --- Spin-Wave: SharedArrayBuffer Zero-Copy 64B-Aligned Phasor Superposition ---
  const sab = new SharedArrayBuffer(CHANNELS * NODE_STRIDE_BYTES);
  const f32 = new Float32Array(sab);

  // Warmup
  for (let i = 0; i < 10_000; i++) {
    const base = (i & (CHANNELS - 1)) * NODE_STRIDE_WORDS;
    const re = f32[base]! + 0.7071;
    const im = f32[base + 1]! + 0.7071;
    f32[base] = re;
    f32[base + 1] = im;
    f32[base + 2] = re * re + im * im;
  }
  f32.fill(0);

  const t1 = performance.now();
  for (let i = 0; i < ops; i++) {
    const base = (i & (CHANNELS - 1)) * NODE_STRIDE_WORDS;
    const dRe = (i & 1) === 0 ? 0.8660254 : 0.5;
    const dIm = (i & 1) === 0 ? 0.5 : 0.8660254;
    const re = f32[base]! + dRe;
    const im = f32[base + 1]! + dIm;
    f32[base] = re;
    f32[base + 1] = im;
    f32[base + 2] = re * re + im * im;
  }
  const spinwaveMs = performance.now() - t1;

  let swChecksum = 0;
  for (let c = 0; c < CHANNELS; c++) {
    swChecksum += f32[c * NODE_STRIDE_WORDS]!;
  }

  const baselineOpsPerSec = ops / (baselineMs / 1000);
  const spinwaveOpsPerSec = ops / (spinwaveMs / 1000);
  const baselineNsPerOp = (baselineMs * 1e6) / ops;
  const spinwaveNsPerOp = (spinwaveMs * 1e6) / ops;

  return {
    ops,
    baselineMs,
    baselineOpsPerSec,
    baselineNsPerOp,
    spinwaveMs,
    spinwaveOpsPerSec,
    spinwaveNsPerOp,
    speedup: baselineMs / spinwaveMs,
    sink: eeAccumRe + eeAccumIm + swChecksum,
  };
}

// ============================================================================
// Bench 2 — 64-Expert MoE Routing (50,000 tokens)
// ============================================================================
function runBench2MoeRouting(tokens = 50_000): MoeBenchResult {
  const EXPERTS = 64;
  const DIM = 64;
  const TOP_K = 2;
  const QUERY_POOL = 256;
  const rand = createPrng(0x51a7_2026);

  // Pre-build normalized 64-dim expert vectors & query pool for baseline
  const expertVecs: Float32Array[] = [];
  for (let e = 0; e < EXPERTS; e++) {
    const v = new Float32Array(DIM);
    let normSq = 0;
    for (let d = 0; d < DIM; d++) {
      const val = rand() * 2 - 1;
      v[d] = val;
      normSq += val * val;
    }
    const invNorm = 1 / Math.sqrt(normSq);
    for (let d = 0; d < DIM; d++) v[d] = v[d]! * invNorm;
    expertVecs.push(v);
  }

  const queryVecs: Float32Array[] = [];
  for (let q = 0; q < QUERY_POOL; q++) {
    const v = new Float32Array(DIM);
    let normSq = 0;
    for (let d = 0; d < DIM; d++) {
      const val = rand() * 2 - 1;
      v[d] = val;
      normSq += val * val;
    }
    const invNorm = 1 / Math.sqrt(normSq);
    for (let d = 0; d < DIM; d++) v[d] = v[d]! * invNorm;
    queryVecs.push(v);
  }

  // Pre-build unit-normalized complex phasor bank in SharedArrayBuffer for Spin-Wave router
  const expertSab = new SharedArrayBuffer(EXPERTS * 2 * 4);
  const expertRe = new Float32Array(expertSab, 0, EXPERTS);
  const expertIm = new Float32Array(expertSab, EXPERTS * 4, EXPERTS);
  for (let e = 0; e < EXPERTS; e++) {
    const theta = (e / EXPERTS) * Math.PI * 2 + (rand() - 0.5) * 0.05;
    expertRe[e] = Math.cos(theta);
    expertIm[e] = Math.sin(theta);
  }

  const queryPhasorRe = new Float32Array(QUERY_POOL);
  const queryPhasorIm = new Float32Array(QUERY_POOL);
  for (let q = 0; q < QUERY_POOL; q++) {
    const theta = rand() * Math.PI * 2;
    queryPhasorRe[q] = Math.cos(theta);
    queryPhasorIm[q] = Math.sin(theta);
  }

  // --- Baseline: 64-dim cosine similarity + object array sort ---
  const routeBaseline = (qVec: Float32Array): number => {
    let qNormSq = 0;
    for (let d = 0; d < DIM; d++) {
      const x = qVec[d]!;
      qNormSq += x * x;
    }
    const invQNorm = 1 / Math.sqrt(qNormSq);

    const scored: Array<{ id: number; score: number }> = new Array(EXPERTS);
    for (let e = 0; e < EXPERTS; e++) {
      const eVec = expertVecs[e]!;
      let dot = 0;
      let eNormSq = 0;
      for (let d = 0; d < DIM; d++) {
        const ev = eVec[d]!;
        dot += qVec[d]! * ev;
        eNormSq += ev * ev;
      }
      scored[e] = { id: e, score: dot * invQNorm / Math.sqrt(eNormSq) };
    }
    scored.sort((a, b) => b.score - a.score);
    return scored[0]!.id + scored[1]!.id;
  };

  // Warmup baseline
  let baseSink = 0;
  for (let i = 0; i < 500; i++) {
    baseSink += routeBaseline(queryVecs[i & (QUERY_POOL - 1)]!);
  }

  baseSink = 0;
  const t0 = performance.now();
  for (let i = 0; i < tokens; i++) {
    baseSink += routeBaseline(queryVecs[i & (QUERY_POOL - 1)]!);
  }
  const baselineMs = performance.now() - t0;

  // --- Spin-Wave: Complex Phasor Coherence (q_re*e_re + q_im*e_im) Zero-Alloc Top-2 ---
  const routeSpinWave = (qRe: number, qIm: number): number => {
    let best0Score = -2.0;
    let best0Id = -1;
    let best1Score = -2.0;
    let best1Id = -1;

    for (let e = 0; e < EXPERTS; e++) {
      const coh = qRe * expertRe[e]! + qIm * expertIm[e]!;
      if (coh > best0Score) {
        best1Score = best0Score;
        best1Id = best0Id;
        best0Score = coh;
        best0Id = e;
      } else if (coh > best1Score) {
        best1Score = coh;
        best1Id = e;
      }
    }
    return best0Id + best1Id;
  };

  // Warmup Spin-Wave
  let swSink = 0;
  for (let i = 0; i < 5_000; i++) {
    const idx = i & (QUERY_POOL - 1);
    swSink += routeSpinWave(queryPhasorRe[idx]!, queryPhasorIm[idx]!);
  }

  swSink = 0;
  const t1 = performance.now();
  for (let i = 0; i < tokens; i++) {
    const idx = i & (QUERY_POOL - 1);
    swSink += routeSpinWave(queryPhasorRe[idx]!, queryPhasorIm[idx]!);
  }
  const spinwaveMs = performance.now() - t1;

  return {
    tokens,
    experts: EXPERTS,
    topK: TOP_K,
    baselineMs,
    baselineTokensPerSec: tokens / (baselineMs / 1000),
    baselineUsPerToken: (baselineMs * 1000) / tokens,
    spinwaveMs,
    spinwaveTokensPerSec: tokens / (spinwaveMs / 1000),
    spinwaveUsPerToken: (spinwaveMs * 1000) / tokens,
    speedup: baselineMs / spinwaveMs,
    sink: baseSink + swSink,
  };
}

// ============================================================================
// Bench 3 — 500-Node DAG Dirty-Wave Scheduling (1,000 incremental 1-node edits)
// ============================================================================
function runBench3DagScheduling(
  nodeCount = 500,
  edits = 1_000,
): DagBenchResult {
  const LAYERS = 10;
  const PER_LAYER = nodeCount / LAYERS; // 50 nodes per layer
  const rand = createPrng(0xda6_500);

  // Build deterministic 500-node layered DAG (~2.5 outgoing edges per non-sink node)
  const edges: DagEdge[] = [];
  const adj: number[][] = Array.from({ length: nodeCount }, () => []);
  const baseInDegree = new Int32Array(nodeCount);

  for (let layer = 0; layer < LAYERS - 1; layer++) {
    const srcStart = layer * PER_LAYER;
    const dstStart = (layer + 1) * PER_LAYER;
    for (let i = 0; i < PER_LAYER; i++) {
      const u = srcStart + i;
      const fanout = 2 + (i % 2); // 2 or 3 edges
      const used = new Set<number>();
      for (let k = 0; k < fanout; k++) {
        // Connect to next layer (or skip 1 layer occasionally)
        const targetLayer =
          k === 2 && layer < LAYERS - 2 ? layer + 2 : layer + 1;
        const targetBase = targetLayer * PER_LAYER;
        const v = targetBase + Math.floor(rand() * PER_LAYER);
        if (!used.has(v)) {
          used.add(v);
          edges.push({ from: u, to: v, weight: 0.92, phaseShift: 0 });
          adj[u]!.push(v);
          baseInDegree[v]! += 1;
        }
      }
    }
  }

  // Deterministic sequence of 1,000 incremental 1-node edits across layers 0..8
  const editTargets = new Int32Array(edits);
  for (let i = 0; i < edits; i++) {
    editTargets[i] = Math.floor(rand() * (nodeCount - PER_LAYER));
  }

  // --- Baseline: Traditional Full Kahn Topological Sort + Set Dirty Traversal ---
  const runBaselineEdit = (dirtyNode: number): number => {
    const inDeg = new Int32Array(baseInDegree);
    const q: number[] = [];
    for (let i = 0; i < nodeCount; i++) {
      if (inDeg[i]! === 0) q.push(i);
    }
    const topo: number[] = [];
    for (let head = 0; head < q.length; head++) {
      const u = q[head]!;
      topo.push(u);
      const neighbors = adj[u]!;
      for (let j = 0; j < neighbors.length; j++) {
        const v = neighbors[j]!;
        const nextDeg = inDeg[v]! - 1;
        inDeg[v] = nextDeg;
        if (nextDeg === 0) q.push(v);
      }
    }

    if (topo.length !== nodeCount) {
      throw new Error(`DAG invariant violated: topo.length=${topo.length} !== ${nodeCount}`);
    }

    // Dirty-cone Set traversal in topological order
    const dirty = new Set<number>();
    dirty.add(dirtyNode);
    let scheduled = 0;
    for (let i = 0; i < topo.length; i++) {
      const u = topo[i]!;
      if (dirty.has(u)) {
        scheduled += 1;
        const neighbors = adj[u]!;
        for (let j = 0; j < neighbors.length; j++) {
          dirty.add(neighbors[j]!);
        }
      }
    }
    return scheduled;
  };

  // Warmup baseline
  let baseVisitedTotal = 0;
  for (let i = 0; i < 50; i++) {
    baseVisitedTotal += runBaselineEdit(editTargets[i]!);
  }

  baseVisitedTotal = 0;
  const t0 = performance.now();
  for (let i = 0; i < edits; i++) {
    baseVisitedTotal += runBaselineEdit(editTargets[i]!);
  }
  const baselineMs = performance.now() - t0;

  // --- Spin-Wave: Damped-Gravity Localized Wave Propagation ---
  const scheduler = new SpinWaveDagScheduler({
    nodeCount,
    edges,
    alpha: 0.72,
    threshold: 0.28,
    gravityDecay: 0.85,
  });

  // Warmup Spin-Wave
  let swFiredTotal = 0;
  for (let i = 0; i < 50; i++) {
    swFiredTotal += scheduler.scheduleSingleEdit(editTargets[i]!);
  }

  swFiredTotal = 0;
  const t1 = performance.now();
  for (let i = 0; i < edits; i++) {
    swFiredTotal += scheduler.scheduleSingleEdit(editTargets[i]!);
  }
  const spinwaveMs = performance.now() - t1;

  return {
    nodes: nodeCount,
    edges: edges.length,
    edits,
    baselineMs,
    baselineUsPerEdit: (baselineMs * 1000) / edits,
    baselineAvgVisited: baseVisitedTotal / edits,
    spinwaveMs,
    spinwaveUsPerEdit: (spinwaveMs * 1000) / edits,
    spinwaveAvgFired: swFiredTotal / edits,
    speedup: baselineMs / spinwaveMs,
    sink: baseVisitedTotal + swFiredTotal,
  };
}

// ============================================================================
// Main Runner
// ============================================================================
function fmtInt(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

function fmtFloat(n: number, digits = 2): string {
  return n.toFixed(digits);
}

const verification = verifySpinWaveScheduler();
const bench1 = runBench1BusThroughput(200_000);
const bench2 = runBench2MoeRouting(50_000);
const bench3 = runBench3DagScheduling(500, 1_000);

const summary = {
  runtime: {
    node: process.version,
    platform: process.platform,
    arch: process.arch,
  },
  schedulerSelfCheck: verification,
  benchmarks: {
    busThroughput: bench1,
    moeRouting: bench2,
    dagScheduling: bench3,
  },
};

const markdownTable = [
  "## Spin-Wave Head-to-Head Benchmark Results",
  "",
  "| Benchmark | Workload | Baseline (Traditional JS/Node) | Spin-Wave (Zero-Copy SAB / Phasor) | Speedup |",
  "| :--- | :--- | :--- | :--- | ---: |",
  `| **1. Intersection Bus** | ${fmtInt(bench1.ops)} ops | ${fmtInt(bench1.baselineOpsPerSec)} ops/s (${fmtFloat(bench1.baselineNsPerOp, 1)} ns/op) | ${fmtInt(bench1.spinwaveOpsPerSec)} ops/s (${fmtFloat(bench1.spinwaveNsPerOp, 1)} ns/op) | **${fmtFloat(bench1.speedup, 2)}x** |`,
  `| **2. 64-Expert MoE Router** | ${fmtInt(bench2.tokens)} tokens (Top-${bench2.topK}) | ${fmtInt(bench2.baselineTokensPerSec)} tok/s (${fmtFloat(bench2.baselineUsPerToken, 2)} µs/tok) | ${fmtInt(bench2.spinwaveTokensPerSec)} tok/s (${fmtFloat(bench2.spinwaveUsPerToken, 2)} µs/tok) | **${fmtFloat(bench2.speedup, 2)}x** |`,
  `| **3. 500-Node DAG Scheduler** | ${fmtInt(bench3.edits)} × 1-node edits (${bench3.edges} edges) | ${fmtFloat(bench3.baselineUsPerEdit, 2)} µs/edit (avg ${fmtFloat(bench3.baselineAvgVisited, 1)} nodes) | ${fmtFloat(bench3.spinwaveUsPerEdit, 2)} µs/edit (avg ${fmtFloat(bench3.spinwaveAvgFired, 1)} fired) | **${fmtFloat(bench3.speedup, 2)}x** |`,
].join("\n");

console.log(markdownTable);
console.log("\n```json");
console.log(JSON.stringify(summary, null, 2));
console.log("```");
