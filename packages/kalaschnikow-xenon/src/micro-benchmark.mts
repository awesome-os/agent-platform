/**
 * @file micro-benchmark.mts
 * @description Minimalist Micro-Benchmark validating the Quaternion SLERP Linear Theory
 * against a Standard Softmax Transformer Baseline across expanding "breathing cycles".
 *
 * Measures:
 *   1. KV-Cache Memory Scaling (O(L) vs O(1) constant)
 *   2. Step Latency across sequence lengths (32 -> 128 -> 512 -> 2048 tokens)
 *   3. Breathing Cycle Dynamics: Inhalation (High Entropy) vs Exhalation (Consolidation)
 *   4. Quaternion S^3 Manifold Unit-Norm Invariance
 *
 * Strictly complies with `erasableSyntaxOnly: true`.
 */

import { Quaternion, SphericalBayesianAccumulator } from './linear-slerp.mts'

export interface BenchmarkStepResult {
  readonly sequenceLength: number
  readonly softmaxKvBytes: number
  readonly linearSlerpBytes: number
  readonly memorySavingsRatio: number
  readonly softmaxStepMs: number
  readonly slerpStepMs: number
  readonly speedupRatio: number
}

export interface BreathingPhaseResult {
  readonly phase: 'inhalation' | 'exhalation'
  readonly tokenStep: number
  readonly entropyBits: number
  readonly bayesianGainAlpha: number
  readonly quaternionNorm: number
  readonly stateConserved: boolean
}

export interface MicroBenchmarkSummary {
  readonly modelDim: number
  readonly steps: BenchmarkStepResult[]
  readonly breathingCycles: BreathingPhaseResult[]
}

/**
 * Standard Softmax Attention Baseline Simulator (Micro Scale).
 */
export class SoftmaxAttentionBaseline {
  public readonly dModel: number
  public readonly keyCache: Float32Array[]
  public readonly valCache: Float32Array[]

  public constructor(dModel: number) {
    this.dModel = dModel
    this.keyCache = []
    this.valCache = []
  }

  public step(q: Float32Array, k: Float32Array, v: Float32Array, out: Float32Array): void {
    // Append to growing KV-cache
    this.keyCache.push(new Float32Array(k))
    this.valCache.push(new Float32Array(v))

    const seqLen = this.keyCache.length
    const scores = new Float32Array(seqLen)
    const scale = 1.0 / Math.sqrt(this.dModel)

    // Compute attention scores over entire past history (O(L) per step)
    let maxScore = -Infinity
    for (let t = 0; t < seqLen; t++) {
      let dot = 0.0
      const pastK = this.keyCache[t]
      for (let i = 0; i < this.dModel; i++) {
        dot += q[i] * pastK[i]
      }
      scores[t] = dot * scale
      if (scores[t] > maxScore) maxScore = scores[t]
    }

    // Softmax normalization
    let sumExp = 0.0
    for (let t = 0; t < seqLen; t++) {
      scores[t] = Math.exp(scores[t] - maxScore)
      sumExp += scores[t]
    }
    const invSum = 1.0 / (sumExp + 1e-12)

    // Weighted value sum
    out.fill(0.0)
    for (let t = 0; t < seqLen; t++) {
      const weight = scores[t] * invSum
      const pastV = this.valCache[t]
      for (let i = 0; i < this.dModel; i++) {
        out[i] += weight * pastV[i]
      }
    }
  }

  public getMemoryBytes(): number {
    // 2 caches (K, V) * seqLen * dModel * 4 bytes
    return this.keyCache.length * this.dModel * 4 * 2
  }

  public reset(): void {
    this.keyCache.length = 0
    this.valCache.length = 0
  }
}

/**
 * Runs the minimal comparative benchmark.
 */
export function runMicroBenchmark(dModel: number = 64): MicroBenchmarkSummary {
  const softmaxModel = new SoftmaxAttentionBaseline(dModel)
  const slerpModel = new SphericalBayesianAccumulator(dModel)

  const seqLengths = [32, 128, 512, 2048]
  const stepResults: BenchmarkStepResult[] = []

  const q = new Float32Array(dModel).fill(0.1)
  const k = new Float32Array(dModel)
  const v = new Float32Array(dModel).fill(0.2)
  const outSoftmax = new Float32Array(dModel)
  const outSlerp = new Float32Array(dModel)

  for (const L of seqLengths) {
    softmaxModel.reset()
    slerpModel.resetContext()

    // 1. Warm-up to sequence length L
    for (let t = 0; t < L - 1; t++) {
      k.fill(Math.sin(t * 0.1))
      softmaxModel.step(q, k, v, outSoftmax)
      slerpModel.step(q, k, v, outSlerp)
    }

    k.fill(Math.sin(L * 0.1))

    // 2. Measure Softmax Step Latency at position L
    const t0Softmax = performance.now()
    const SOFTMAX_REPS = 50
    for (let rep = 0; rep < SOFTMAX_REPS; rep++) {
      softmaxModel.step(q, k, v, outSoftmax)
      // Pop last addition to keep length at L
      softmaxModel.keyCache.pop()
      softmaxModel.valCache.pop()
    }
    const softmaxStepMs = (performance.now() - t0Softmax) / SOFTMAX_REPS

    // 3. Measure Linear SLERP Step Latency at position L
    const t0Slerp = performance.now()
    const SLERP_REPS = 50
    for (let rep = 0; rep < SLERP_REPS; rep++) {
      slerpModel.step(q, k, v, outSlerp)
    }
    const slerpStepMs = (performance.now() - t0Slerp) / SLERP_REPS

    const softmaxBytes = softmaxModel.getMemoryBytes()
    const slerpBytes = slerpModel.getMetrics().memoryFootprintBytes

    stepResults.push({
      sequenceLength: L,
      softmaxKvBytes: softmaxBytes,
      linearSlerpBytes: slerpBytes,
      memorySavingsRatio: Math.round((softmaxBytes / slerpBytes) * 100) / 100,
      softmaxStepMs: Math.round(softmaxStepMs * 1000) / 1000,
      slerpStepMs: Math.round(slerpStepMs * 1000) / 1000,
      speedupRatio: Math.round((softmaxStepMs / Math.max(0.0001, slerpStepMs)) * 100) / 100,
    })
  }

  // 4. Simulate the Nature Reflection "Breathing Cycles"
  // Cycle 1: Inhalation (High Entropy / Novelty) -> Exhalation (Low Entropy / Consolidation)
  // Cycle 2: Inhalation -> Exhalation
  const breathingCycles: BreathingPhaseResult[] = []
  slerpModel.resetContext()

  for (let cycle = 0; cycle < 2; cycle++) {
    // Inhalation: High Entropy noise tokens
    for (let step = 0; step < 5; step++) {
      for (let i = 0; i < dModel; i++) {
        k[i] = (Math.random() - 0.5) * 4.0
      }
      slerpModel.step(q, k, v, outSlerp)

      const q0 = slerpModel.stateQuats[0]
      const norm = Math.sqrt(q0.w * q0.w + q0.x * q0.x + q0.y * q0.y + q0.z * q0.z)
      const metrics = slerpModel.getMetrics()

      breathingCycles.push({
        phase: 'inhalation',
        tokenStep: cycle * 10 + step,
        entropyBits: Math.round(metrics.averageEntropy * 100) / 100,
        bayesianGainAlpha: Math.round(metrics.lastSurpriseGain * 1000) / 1000,
        quaternionNorm: Math.round(norm * 100000) / 100000,
        stateConserved: Math.abs(norm - 1.0) < 1e-5,
      })
    }

    // Exhalation: Low Entropy predictable repeating tokens (resting phase)
    for (let step = 0; step < 5; step++) {
      k.fill(0.05) // Smooth predictable tokens
      slerpModel.step(q, k, v, outSlerp)

      const q0 = slerpModel.stateQuats[0]
      const norm = Math.sqrt(q0.w * q0.w + q0.x * q0.x + q0.y * q0.y + q0.z * q0.z)
      const metrics = slerpModel.getMetrics()

      breathingCycles.push({
        phase: 'exhalation',
        tokenStep: cycle * 10 + 5 + step,
        entropyBits: Math.round(metrics.averageEntropy * 100) / 100,
        bayesianGainAlpha: Math.round(metrics.lastSurpriseGain * 1000) / 1000,
        quaternionNorm: Math.round(norm * 100000) / 100000,
        stateConserved: Math.abs(norm - 1.0) < 1e-5,
      })
    }
  }

  return {
    modelDim: dModel,
    steps: stepResults,
    breathingCycles,
  }
}
