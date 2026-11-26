/**
 * Spin-Wave + Damped-Gravity DAG Scheduler (`erasableSyntaxOnly: true`, zero external deps).
 *
 * Each node in the build/intersection DAG owns a 64-byte cache-line-aligned slot in a
 * `SharedArrayBuffer` (`B8G_ARENA_V1`-compatible layout) storing its complex phasor
 * `(re, im)`, instantaneous wave `energy`, damping/gravity parameters, and monotonic `epoch`.
 *
 * Physics model:
 * - Dirty leaves inject wave energy: `re += A * cos(phi)`, `im += A * sin(phi)`.
 * - Waves propagate along directed DAG edges with damping `alpha < 1.0` and optional edge
 *   phase shift `deltaPhi` (enabling constructive and destructive interference).
 * - A node fires when its constructive interference amplitude exceeds `threshold`.
 * - Stale priorities decay automatically via damped-gravity attenuation `alpha^(dt)` on touch
 *   or explicit `decayTick()`.
 */

/** 64-byte cache-line stride per node (16 x 32-bit words). */
export const NODE_STRIDE_BYTES = 64;
export const NODE_STRIDE_WORDS = 16;

// Word offsets within each 64-byte node slot:
const OFF_RE = 0; // f32: real phasor component
const OFF_IM = 1; // f32: imaginary phasor component
const OFF_ENERGY = 2; // f32: instantaneous amplitude sqrt(re^2 + im^2)
const OFF_PHASE = 3; // f32: intrinsic node phase shift (radians)
const OFF_DAMPING = 4; // f32: local damping multiplier (default 1.0)
const OFF_STATE = 5; // i32: 0 = idle, 1 = queued, 2 = fired
const OFF_EPOCH = 6; // i32: last active wave epoch (for O(1) lazy stale decay)
const OFF_PRIORITY = 7; // f32: gravity-damped priority score

export interface DagEdge {
  readonly from: number;
  readonly to: number;
  /** Optional edge coupling weight (default: 1.0). */
  readonly weight?: number;
  /** Optional edge phase shift in radians (default: 0.0). */
  readonly phaseShift?: number;
}

export interface SpinWaveSchedulerOptions {
  readonly nodeCount: number;
  readonly edges: readonly DagEdge[];
  /** Global per-hop damping factor alpha in (0, 1). Default: 0.72. */
  readonly alpha?: number;
  /** Constructive interference amplitude threshold to fire a node. Default: 0.25. */
  readonly threshold?: number;
  /** Gravity decay factor applied per epoch gap to stale priorities. Default: 0.85. */
  readonly gravityDecay?: number;
}

export interface NodePhasorSnapshot {
  readonly nodeId: number;
  readonly re: number;
  readonly im: number;
  readonly energy: number;
  readonly priority: number;
  readonly state: number;
  readonly epoch: number;
}

export interface SchedulerVerificationResult {
  readonly ok: boolean;
  readonly alignmentBytes: number;
  readonly constructiveFired: boolean;
  readonly destructiveCancelled: boolean;
  readonly dampedStopCount: number;
  readonly decayedPriorityRatio: number;
}

export class SpinWaveDagScheduler {
  readonly nodeCount: number;
  readonly alpha: number;
  readonly threshold: number;
  readonly gravityDecay: number;

  /** 64-byte-aligned SharedArrayBuffer holding all node phasors. */
  readonly buffer: SharedArrayBuffer;
  private readonly f32: Float32Array;
  private readonly i32: Int32Array;

  // Flat CSR adjacency arrays for zero-allocation propagation
  private readonly edgeOffsets: Int32Array;
  private readonly edgeTargets: Int32Array;
  private readonly edgeWeights: Float32Array;
  private readonly edgeCos: Float32Array;
  private readonly edgeSin: Float32Array;

  // Pre-allocated worklists (zero GC allocations during propagate)
  private readonly queue: Int32Array;
  private readonly firedScratch: Int32Array;
  private queueHead = 0;
  private queueTail = 0;
  private currentEpoch = 1;

  constructor(options: SpinWaveSchedulerOptions) {
    const {
      nodeCount,
      edges,
      alpha = 0.72,
      threshold = 0.25,
      gravityDecay = 0.85,
    } = options;

    if (nodeCount <= 0) {
      throw new RangeError(`nodeCount must be > 0, got ${nodeCount}`);
    }
    if (!(alpha > 0 && alpha < 1)) {
      throw new RangeError(`alpha must be in (0, 1), got ${alpha}`);
    }

    this.nodeCount = nodeCount;
    this.alpha = alpha;
    this.threshold = threshold;
    this.gravityDecay = gravityDecay;

    // Allocate 64-byte-aligned SharedArrayBuffer arena
    this.buffer = new SharedArrayBuffer(nodeCount * NODE_STRIDE_BYTES);
    this.f32 = new Float32Array(this.buffer);
    this.i32 = new Int32Array(this.buffer);

    // Initialize default per-node damping = 1.0
    for (let i = 0; i < nodeCount; i++) {
      this.f32[i * NODE_STRIDE_WORDS + OFF_DAMPING] = 1.0;
    }

    // Build flat CSR adjacency representation
    const outDegree = new Int32Array(nodeCount);
    for (let i = 0; i < edges.length; i++) {
      const from = edges[i]!.from;
      if (from >= 0 && from < nodeCount) {
        outDegree[from]! += 1;
      }
    }

    this.edgeOffsets = new Int32Array(nodeCount + 1);
    let totalEdges = 0;
    for (let i = 0; i < nodeCount; i++) {
      this.edgeOffsets[i] = totalEdges;
      totalEdges += outDegree[i]!;
    }
    this.edgeOffsets[nodeCount] = totalEdges;

    this.edgeTargets = new Int32Array(totalEdges);
    this.edgeWeights = new Float32Array(totalEdges);
    this.edgeCos = new Float32Array(totalEdges);
    this.edgeSin = new Float32Array(totalEdges);

    const writeCursor = new Int32Array(this.edgeOffsets.subarray(0, nodeCount));
    for (let i = 0; i < edges.length; i++) {
      const e = edges[i]!;
      if (e.from < 0 || e.from >= nodeCount || e.to < 0 || e.to >= nodeCount) {
        continue;
      }
      const pos = writeCursor[e.from]!;
      writeCursor[e.from] = pos + 1;
      this.edgeTargets[pos] = e.to;
      this.edgeWeights[pos] = e.weight ?? 1.0;
      const phi = e.phaseShift ?? 0.0;
      this.edgeCos[pos] = Math.cos(phi);
      this.edgeSin[pos] = Math.sin(phi);
    }

    // Ring buffer sized to 2x nodeCount (power-of-2 not required; simple linear reset per wave)
    this.queue = new Int32Array(Math.max(64, nodeCount * 2));
    this.firedScratch = new Int32Array(nodeCount);
  }

  /** Advance the monotonic wave epoch (O(1) lazy invalidation + stale gravity decay). */
  beginEpoch(): number {
    this.currentEpoch += 1;
    this.queueHead = 0;
    this.queueTail = 0;
    return this.currentEpoch;
  }

  /**
   * Lazily synchronizes a node slot to `currentEpoch`, decaying any stale priority
   * by `gravityDecay^(currentEpoch - lastEpoch)` and clearing stale wave phasors.
   */
  private syncNodeEpoch(nodeId: number, epoch: number): number {
    const base = nodeId * NODE_STRIDE_WORDS;
    const prevEpoch = this.i32[base + OFF_EPOCH]!;
    if (prevEpoch !== epoch) {
      const dt = epoch - prevEpoch;
      if (prevEpoch > 0 && dt > 0) {
        const oldPriority = this.f32[base + OFF_PRIORITY]!;
        if (oldPriority > 1e-6) {
          this.f32[base + OFF_PRIORITY] =
            oldPriority * Math.pow(this.gravityDecay, dt);
        } else {
          this.f32[base + OFF_PRIORITY] = 0;
        }
      }
      this.f32[base + OFF_RE] = 0;
      this.f32[base + OFF_IM] = 0;
      this.f32[base + OFF_ENERGY] = 0;
      this.i32[base + OFF_STATE] = 0;
      this.i32[base + OFF_EPOCH] = epoch;
    }
    return base;
  }

  /**
   * Injects spin-wave energy into a dirty leaf (or any node) using `(amplitude, phase)`.
   * Multiple injections into the same node in the same epoch superpose coherently.
   */
  inject(nodeId: number, amplitude = 1.0, phase = 0.0): number {
    const epoch = this.currentEpoch;
    const base = this.syncNodeEpoch(nodeId, epoch);

    const addRe = phase === 0.0 ? amplitude : amplitude * Math.cos(phase);
    const addIm = phase === 0.0 ? 0.0 : amplitude * Math.sin(phase);

    const re = this.f32[base + OFF_RE]! + addRe;
    const im = this.f32[base + OFF_IM]! + addIm;
    const energy = Math.sqrt(re * re + im * im);

    this.f32[base + OFF_RE] = re;
    this.f32[base + OFF_IM] = im;
    this.f32[base + OFF_ENERGY] = energy;

    if (energy >= this.threshold && this.i32[base + OFF_STATE]! === 0) {
      this.i32[base + OFF_STATE] = 1;
      if (this.queueTail < this.queue.length) {
        this.queue[this.queueTail++] = nodeId;
      }
    }
    return energy;
  }

  /**
   * Propagates all queued spin-wave fronts along DAG edges with damping `alpha < 1.0`.
   * Nodes fire when constructive interference exceeds `threshold`.
   *
   * Returns the number of fired nodes; the fired node IDs reside in `this.getFiredView(count)`.
   */
  propagateCount(): number {
    const f32 = this.f32;
    const i32 = this.i32;
    const edgeOffsets = this.edgeOffsets;
    const edgeTargets = this.edgeTargets;
    const edgeWeights = this.edgeWeights;
    const edgeCos = this.edgeCos;
    const edgeSin = this.edgeSin;
    const queue = this.queue;
    const fired = this.firedScratch;

    const alpha = this.alpha;
    const threshold = this.threshold;
    const thresholdSq = threshold * threshold;
    const epoch = this.currentEpoch;
    const gravityDecay = this.gravityDecay;

    let firedCount = 0;

    while (this.queueHead < this.queueTail) {
      const u = queue[this.queueHead++]!;
      const uBase = u * NODE_STRIDE_WORDS;

      // Re-check energy in case destructive interference lowered it before firing
      const uRe = f32[uBase + OFF_RE]!;
      const uIm = f32[uBase + OFF_IM]!;
      const uEnergySq = uRe * uRe + uIm * uIm;
      if (uEnergySq < thresholdSq) {
        i32[uBase + OFF_STATE] = 0;
        continue;
      }

      // Mark node as fired and update gravity priority
      i32[uBase + OFF_STATE] = 2;
      const uEnergy = Math.sqrt(uEnergySq);
      f32[uBase + OFF_ENERGY] = uEnergy;
      f32[uBase + OFF_PRIORITY] = f32[uBase + OFF_PRIORITY]! + uEnergy;
      fired[firedCount++] = u;

      // Damped wave emission along outgoing DAG edges
      const dampedAlpha = alpha * f32[uBase + OFF_DAMPING]!;
      const start = edgeOffsets[u]!;
      const end = edgeOffsets[u + 1]!;

      for (let e = start; e < end; e++) {
        const v = edgeTargets[e]!;
        const vBase = v * NODE_STRIDE_WORDS;

        // Inline lazy epoch sync for hot-loop speed
        const prevEpoch = i32[vBase + OFF_EPOCH]!;
        if (prevEpoch !== epoch) {
          const dt = epoch - prevEpoch;
          if (prevEpoch > 0 && dt > 0) {
            const oldP = f32[vBase + OFF_PRIORITY]!;
            f32[vBase + OFF_PRIORITY] =
              oldP > 1e-6 ? oldP * Math.pow(gravityDecay, dt) : 0;
          }
          f32[vBase + OFF_RE] = 0;
          f32[vBase + OFF_IM] = 0;
          f32[vBase + OFF_ENERGY] = 0;
          i32[vBase + OFF_STATE] = 0;
          i32[vBase + OFF_EPOCH] = epoch;
        }

        const scale = dampedAlpha * edgeWeights[e]!;
        const c = edgeCos[e]!;
        const s = edgeSin[e]!;
        // Complex rotation + damping: (uRe + i*uIm) * scale * (c + i*s)
        const dRe = scale * (uRe * c - uIm * s);
        const dIm = scale * (uRe * s + uIm * c);

        const vRe = f32[vBase + OFF_RE]! + dRe;
        const vIm = f32[vBase + OFF_IM]! + dIm;
        f32[vBase + OFF_RE] = vRe;
        f32[vBase + OFF_IM] = vIm;

        const vEnergySq = vRe * vRe + vIm * vIm;
        if (vEnergySq >= thresholdSq && i32[vBase + OFF_STATE]! === 0) {
          f32[vBase + OFF_ENERGY] = Math.sqrt(vEnergySq);
          i32[vBase + OFF_STATE] = 1;
          if (this.queueTail < queue.length) {
            queue[this.queueTail++] = v;
          }
        }
      }
    }

    return firedCount;
  }

  /** Zero-copy subarray view of the fired node IDs from the last `propagateCount()` call. */
  getFiredView(count: number): Int32Array {
    return this.firedScratch.subarray(0, count);
  }

  /** Convenience helper: starts a new epoch, injects a single dirty node, and propagates. */
  scheduleSingleEdit(nodeId: number, amplitude = 1.0, phase = 0.0): number {
    this.beginEpoch();
    this.inject(nodeId, amplitude, phase);
    return this.propagateCount();
  }

  /** Reads a node's current 64B-aligned phasor state (applying lazy stale decay). */
  inspectNode(nodeId: number): NodePhasorSnapshot {
    const base = this.syncNodeEpoch(nodeId, this.currentEpoch);
    return {
      nodeId,
      re: this.f32[base + OFF_RE]!,
      im: this.f32[base + OFF_IM]!,
      energy: this.f32[base + OFF_ENERGY]!,
      priority: this.f32[base + OFF_PRIORITY]!,
      state: this.i32[base + OFF_STATE]!,
      epoch: this.i32[base + OFF_EPOCH]!,
    };
  }
}

/**
 * Self-check verifying:
 * 1. 64-byte SharedArrayBuffer alignment (`NODE_STRIDE_BYTES === 64`).
 * 2. Constructive interference fires a target node that a single sub-threshold wave cannot.
 * 3. Destructive interference (pi phase shift) cancels out and prevents firing.
 * 4. Damped-gravity attenuation halts propagation on a long chain.
 * 5. Stale priority decays automatically across epochs.
 */
export function verifySpinWaveScheduler(): SchedulerVerificationResult {
  // 1 & 2: Constructive interference: nodes 0 and 1 -> node 2
  const constructive = new SpinWaveDagScheduler({
    nodeCount: 3,
    edges: [
      { from: 0, to: 2, weight: 0.6, phaseShift: 0 },
      { from: 1, to: 2, weight: 0.6, phaseShift: 0 },
    ],
    alpha: 0.8,
    threshold: 0.75,
  });

  // Single leaf -> 1.0 * 0.8 * 0.6 = 0.48 < 0.75 (node 2 should NOT fire)
  constructive.beginEpoch();
  constructive.inject(0, 1.0, 0.0);
  const singleCount = constructive.propagateCount();
  const singleFired = constructive.getFiredView(singleCount);
  if (singleFired.includes(2)) {
    throw new Error("Self-check failed: single sub-threshold wave fired node 2");
  }

  // Two in-phase leaves -> 0.48 + 0.48 = 0.96 >= 0.75 (node 2 MUST fire)
  constructive.beginEpoch();
  constructive.inject(0, 1.0, 0.0);
  constructive.inject(1, 1.0, 0.0);
  const dualCount = constructive.propagateCount();
  const dualFired = constructive.getFiredView(dualCount);
  const constructiveFired = dualFired.includes(2);
  if (!constructiveFired) {
    throw new Error("Self-check failed: constructive interference did not fire node 2");
  }

  // 3: Destructive interference: node 0 (phase 0) + node 1 (phase pi) -> cancels to 0
  const destructive = new SpinWaveDagScheduler({
    nodeCount: 3,
    edges: [
      { from: 0, to: 2, weight: 0.9, phaseShift: 0 },
      { from: 1, to: 2, weight: 0.9, phaseShift: Math.PI },
    ],
    alpha: 0.9,
    threshold: 0.5,
  });
  destructive.beginEpoch();
  destructive.inject(0, 1.0, 0.0);
  destructive.inject(1, 1.0, 0.0);
  const destCount = destructive.propagateCount();
  const destFired = destructive.getFiredView(destCount);
  const destructiveCancelled = !destFired.includes(2);
  if (!destructiveCancelled) {
    throw new Error("Self-check failed: destructive interference failed to cancel node 2");
  }

  // 4 & 5: Damped-gravity attenuation on a 20-node chain + automatic stale priority decay
  const chainEdges: DagEdge[] = [];
  for (let i = 0; i < 19; i++) {
    chainEdges.push({ from: i, to: i + 1 });
  }
  const chain = new SpinWaveDagScheduler({
    nodeCount: 20,
    edges: chainEdges,
    alpha: 0.65,
    threshold: 0.2,
    gravityDecay: 0.5,
  });
  const dampedStopCount = chain.scheduleSingleEdit(0, 1.0, 0.0);
  if (dampedStopCount <= 1 || dampedStopCount >= 20) {
    throw new Error(
      `Self-check failed: expected localized damped propagation in (1, 20), got ${dampedStopCount}`,
    );
  }

  const pBefore = chain.inspectNode(0).priority;
  chain.beginEpoch();
  chain.beginEpoch(); // +2 epochs -> priority should decay by 0.5^2 = 0.25
  const pAfter = chain.inspectNode(0).priority;
  const decayedPriorityRatio = pBefore > 0 ? pAfter / pBefore : 0;
  if (Math.abs(decayedPriorityRatio - 0.25) > 1e-4) {
    throw new Error(
      `Self-check failed: expected stale priority decay ratio 0.25, got ${decayedPriorityRatio}`,
    );
  }

  const alignmentBytes = chain.buffer.byteLength / chain.nodeCount;
  if (alignmentBytes !== NODE_STRIDE_BYTES) {
    throw new Error(
      `Self-check failed: expected 64B alignment, got ${alignmentBytes}`,
    );
  }

  return {
    ok: true,
    alignmentBytes,
    constructiveFired,
    destructiveCancelled,
    dampedStopCount,
    decayedPriorityRatio,
  };
}
