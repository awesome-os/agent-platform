/**
 * @file intersections/b8g-llm-studio/lora-engine.mts
 * @description Zero-Dependency Sovereign LoRA / Unsloth-Style Embedding & Fine-Tune Engine.
 *
 * Pure Node 26 TypeScript:
 *   - Zero external npm dependencies
 *   - Strict `erasableSyntaxOnly: true` (no enums, no namespaces, no parameter properties)
 *   - Zero `node:http`
 *   - Real Float32 linear algebra, Matryoshka L2-renormalized slicing, analytical InfoNCE/triplet
 *     contrastive backprop through L2-norm + low-rank `(A, B)` adapters, exact `W' = W + (α/r)·B·A`
 *     weight merging with SHA-256 digests, and pure Web `ReadableStream` training telemetry.
 */

import { createHash } from 'node:crypto'
import {
  createLivenessTransform,
  type LivenessFrame,
} from '../schema/index.mts'

export type TaskAdapterId =
  | 'retrieval.query'
  | 'retrieval.passage'
  | 'text-matching'
  | 'error-repair'
  | 'code-search'

export const TASK_ADAPTER_IDS: readonly TaskAdapterId[] = [
  'retrieval.query',
  'retrieval.passage',
  'text-matching',
  'error-repair',
  'code-search',
]

export type LoraStrategy = 'merge-then-stack' | 'direct-task-finetune'

export type QuantizationKind = 'nf4' | 'q8' | 'fp16'

export interface ContrastivePair {
  readonly anchor: string
  readonly positive: string
  readonly negative: string
}

export interface LoraAdapter {
  readonly id: string
  readonly taskId: TaskAdapterId
  strategy: LoraStrategy
  readonly dim: number
  readonly rank: number
  readonly alpha: number
  /** Row-major `[rank x dim]` Float32 matrix. */
  readonly A: Float32Array
  /** Row-major `[dim x rank]` Float32 matrix. */
  readonly B: Float32Array
  sha256: string
  readonly lossHistory: number[]
  step: number
  readonly createdAt: string
  updatedAt: string
}

export interface LoraAdapterSummary {
  readonly id: string
  readonly taskId: TaskAdapterId
  readonly strategy: LoraStrategy
  readonly dim: number
  readonly rank: number
  readonly alpha: number
  readonly sha256: string
  readonly step: number
  readonly lastLoss: number | null
  readonly lossHistory: readonly number[]
  readonly paramCount: number
  readonly bytes: number
}

export interface TrainStepResult {
  readonly step: number
  readonly loss: number
  readonly posSim: number
  readonly negSim: number
  readonly gradNorm: number
  readonly adapterSha256: string
}

export interface TrainStepEvent {
  readonly step: number
  readonly totalSteps: number
  readonly loss: number
  readonly posSim: number
  readonly negSim: number
  readonly lr: number
  readonly elapsedMs: number
  readonly adapterSha256: string
}

export interface LoraTrainStreamOptions {
  readonly adapter: LoraAdapter
  readonly pairs: readonly ContrastivePair[]
  readonly steps?: number
  readonly lr?: number
  readonly baseW?: Float32Array
  readonly temperature?: number
}

export interface MergeLoraResult {
  readonly mergedWeights: Float32Array
  readonly sha256: string
  readonly bytes: number
  readonly dim: number
}

export interface UnslothMemoryEstimate {
  readonly paramsBillion: number
  readonly rank: number
  readonly quant: QuantizationKind
  readonly batchSize: number
  readonly baseMb: number
  readonly loraMb: number
  readonly optimizerMb: number
  readonly totalMb: number
  readonly savingsPctVsFp16: number
}

export interface StrategyApplyResult {
  readonly strategy: LoraStrategy
  readonly taskId: TaskAdapterId
  readonly baseSha256: string
  readonly adapter: LoraAdapter
  readonly merged: MergeLoraResult | null
  readonly extraInferenceLatency: boolean
}

export interface LoraStudioEngineOptions {
  readonly dim?: number
  readonly defaultRank?: number
  readonly defaultAlpha?: number
  readonly baseWeights?: Float32Array
}

// ---------------------------------------------------------------------------
// Deterministic Hashing, PRNG & SHA-256 Digest Utilities
// ---------------------------------------------------------------------------

function fnv1a32(input: string, seed = 0x811c9dc5): number {
  let hash = seed >>> 0
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193) >>> 0
  }
  return hash >>> 0
}

function createMulberry32(seed: number): () => number {
  let state = seed >>> 0
  return (): number => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = Math.imul(state ^ (state >>> 15), 1 | state)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function computeFloat32Sha256(data: Float32Array): string {
  const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
  return createHash('sha256').update(view).digest('hex')
}

export function computeAdapterDigest(
  A: Float32Array,
  B: Float32Array,
  taskId: TaskAdapterId,
  rank: number,
  alpha: number
): string {
  const hash = createHash('sha256')
  hash.update(`${taskId}|r=${rank}|a=${alpha}|`)
  hash.update(new Uint8Array(A.buffer, A.byteOffset, A.byteLength))
  hash.update(new Uint8Array(B.buffer, B.byteOffset, B.byteLength))
  return hash.digest('hex')
}

// ---------------------------------------------------------------------------
// Core Float32 Vector & Matrix Primitives
// ---------------------------------------------------------------------------

export function createIdentityMatrix(dim: number): Float32Array {
  if (!Number.isInteger(dim) || dim < 1) {
    throw new RangeError(`dim must be a positive integer, got ${dim}`)
  }
  const W = new Float32Array(dim * dim)
  for (let i = 0; i < dim; i++) {
    W[i * dim + i] = 1.0
  }
  return W
}

export function l2Norm(vec: Float32Array): number {
  let sum = 0
  for (let i = 0; i < vec.length; i++) {
    const v = vec[i]
    sum += v * v
  }
  return Math.sqrt(sum)
}

export function normalizeL2(vec: Float32Array): Float32Array {
  const out = new Float32Array(vec.length)
  const norm = l2Norm(vec)
  if (norm <= 1e-12) {
    if (out.length > 0) {
      out[0] = 1.0
    }
    return out
  }
  const inv = 1.0 / norm
  for (let i = 0; i < vec.length; i++) {
    out[i] = vec[i] * inv
  }
  return out
}

/**
 * Deterministic lightweight text-to-vector base encoder (`dim = 64` by default).
 * Uses signed feature hashing over normalized words and character trigrams,
 * followed by L2 normalization so output vectors lie on the unit hypersphere.
 */
export function encodeBase(text: string, dim = 64): Float32Array {
  if (!Number.isInteger(dim) || dim < 1) {
    throw new RangeError(`dim must be a positive integer, got ${dim}`)
  }
  const vec = new Float32Array(dim)
  const normalized = text.toLowerCase().trim()

  if (normalized.length === 0) {
    const seed = fnv1a32('<empty>')
    vec[seed % dim] = 1.0
    return vec
  }

  const tokens = normalized.split(/[^a-z0-9_./:-]+/).filter((t) => t.length > 0)
  for (let tIdx = 0; tIdx < tokens.length; tIdx++) {
    const tok = tokens[tIdx]
    // Whole-token signed projections (2 independent hashes per token)
    const h1 = fnv1a32(tok, 0x811c9dc5)
    const h2 = fnv1a32(tok, 0x9e3779b1)
    const idx1 = h1 % dim
    const idx2 = h2 % dim
    vec[idx1] += (h1 & 0x80000000) === 0 ? 1.5 : -1.5
    vec[idx2] += (h2 & 0x80000000) === 0 ? 1.0 : -1.0

    // Character trigrams for subword & typo resilience
    const padded = `^${tok}$`
    for (let i = 0; i <= padded.length - 3; i++) {
      const tri = padded.slice(i, i + 3)
      const ht = fnv1a32(tri, 0x517cc1b7)
      const idx = ht % dim
      vec[idx] += (ht & 0x80000000) === 0 ? 0.45 : -0.45
    }
  }

  // Bigram phrase features for contextual ordering
  for (let i = 0; i < tokens.length - 1; i++) {
    const bigram = `${tokens[i]}_${tokens[i + 1]}`
    const hb = fnv1a32(bigram, 0x27d4eb2f)
    vec[hb % dim] += (hb & 0x80000000) === 0 ? 0.85 : -0.85
  }

  return normalizeL2(vec)
}

/**
 * Matryoshka Representation Learning (MRL) dimension truncation with L2 renormalization.
 */
export function sliceMatryoshka(vec: Float32Array, targetDim: number): Float32Array {
  if (!Number.isInteger(targetDim) || targetDim < 1 || targetDim > vec.length) {
    throw new RangeError(
      `targetDim must be an integer in [1, ${vec.length}], got ${targetDim}`
    )
  }
  const sliced = vec.subarray(0, targetDim)
  return normalizeL2(sliced)
}

/**
 * Exact cosine similarity between two equal-length Float32 vectors.
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new RangeError(
      `Vector dimension mismatch: ${a.length} !== ${b.length}`
    )
  }
  let dot = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < a.length; i++) {
    const av = a[i]
    const bv = b[i]
    dot += av * bv
    normA += av * av
    normB += bv * bv
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB)
  if (denom <= 1e-12) {
    return 0
  }
  return Math.max(-1, Math.min(1, dot / denom))
}

/**
 * Computes `y = W * x + (alpha / rank) * B * (A * x)` (pre-normalization)
 * along with intermediate `u = A * x` needed for exact backpropagation.
 */
function forwardWithLora(
  x: Float32Array,
  baseW: Float32Array,
  adapter: LoraAdapter
): { readonly y: Float32Array; readonly z: Float32Array; readonly r: number; readonly u: Float32Array } {
  const { dim, rank, alpha, A, B } = adapter
  const scale = alpha / rank

  // u = A * x  (rank)
  const u = new Float32Array(rank)
  for (let k = 0; k < rank; k++) {
    let sum = 0
    const rowOff = k * dim
    for (let j = 0; j < dim; j++) {
      sum += A[rowOff + j] * x[j]
    }
    u[k] = sum
  }

  // y = W * x + scale * B * u  (dim)
  const y = new Float32Array(dim)
  for (let i = 0; i < dim; i++) {
    let sum = 0
    const wOff = i * dim
    for (let j = 0; j < dim; j++) {
      sum += baseW[wOff + j] * x[j]
    }
    let loraDelta = 0
    const bOff = i * rank
    for (let k = 0; k < rank; k++) {
      loraDelta += B[bOff + k] * u[k]
    }
    y[i] = sum + scale * loraDelta
  }

  const r = Math.max(l2Norm(y), 1e-8)
  const z = new Float32Array(dim)
  const invR = 1.0 / r
  for (let i = 0; i < dim; i++) {
    z[i] = y[i] * invR
  }

  return { y, z, r, u }
}

/**
 * Applies `y = W * x` (no LoRA) and L2-normalizes.
 */
export function applyLinearMatrix(W: Float32Array, x: Float32Array, dim: number): Float32Array {
  const y = new Float32Array(dim)
  for (let i = 0; i < dim; i++) {
    let sum = 0
    const rowOff = i * dim
    for (let j = 0; j < dim; j++) {
      sum += W[rowOff + j] * x[j]
    }
    y[i] = sum
  }
  return y
}

// ---------------------------------------------------------------------------
// Adapter Factory, Contrastive Training & Exact Weight Merging
// ---------------------------------------------------------------------------

export function createLoraAdapter(opts: {
  readonly taskId: TaskAdapterId
  readonly dim?: number
  readonly rank?: number
  readonly alpha?: number
  readonly strategy?: LoraStrategy
  readonly seedSalt?: string
}): LoraAdapter {
  const dim = opts.dim ?? 64
  const rank = opts.rank ?? 8
  const alpha = opts.alpha ?? rank * 2
  const strategy: LoraStrategy = opts.strategy ?? 'direct-task-finetune'

  if (!Number.isInteger(dim) || dim < 1) {
    throw new RangeError(`dim must be >= 1, got ${dim}`)
  }
  if (!Number.isInteger(rank) || rank < 1 || rank > dim) {
    throw new RangeError(`rank must be an integer in [1, ${dim}], got ${rank}`)
  }

  const A = new Float32Array(rank * dim)
  const B = new Float32Array(dim * rank) // Zero-initialized per LoRA spec (Hu et al.)

  const rng = createMulberry32(
    fnv1a32(`${opts.taskId}:${dim}:${rank}:${opts.seedSalt ?? 'v1'}`)
  )
  const bound = Math.sqrt(1.0 / dim)
  for (let i = 0; i < A.length; i++) {
    A[i] = (rng() * 2 - 1) * bound
  }

  const now = new Date().toISOString()
  const sha256 = computeAdapterDigest(A, B, opts.taskId, rank, alpha)

  return {
    id: `lora-${opts.taskId}-r${rank}`,
    taskId: opts.taskId,
    strategy,
    dim,
    rank,
    alpha,
    A,
    B,
    sha256,
    lossHistory: [],
    step: 0,
    createdAt: now,
    updatedAt: now,
  }
}

/**
 * Performs one real analytical gradient descent step on a contrastive triplet/InfoNCE
 * objective:
 *   `L = log(1 + exp((cos(z_a, z_n) - cos(z_a, z_p)) / tau))`
 * backpropagated exactly through L2-normalization and the low-rank `(A, B)` adapter.
 */
export function trainContrastiveStep(
  adapter: LoraAdapter,
  pairs: readonly ContrastivePair[],
  lr: number,
  opts?: {
    readonly baseW?: Float32Array
    readonly temperature?: number
  }
): TrainStepResult {
  if (pairs.length === 0) {
    throw new RangeError('trainContrastiveStep requires at least one contrastive pair')
  }
  if (!(lr > 0) || !Number.isFinite(lr)) {
    throw new RangeError(`lr must be a positive finite number, got ${lr}`)
  }

  const { dim, rank, alpha, A, B } = adapter
  const baseW = opts?.baseW ?? createIdentityMatrix(dim)
  const tau = opts?.temperature ?? 0.25
  const scale = alpha / rank

  const gradA = new Float32Array(rank * dim)
  const gradB = new Float32Array(dim * rank)
  const dyBuf = new Float32Array(dim)
  const duBuf = new Float32Array(rank)

  let totalLoss = 0
  let totalPosSim = 0
  let totalNegSim = 0

  const accumulateSampleGrad = (
    dLdz: Float32Array,
    z: Float32Array,
    r: number,
    u: Float32Array,
    x: Float32Array
  ): void => {
    // Exact Jacobian of L2 normalization: dL/dy = (dL/dz - dot(dL/dz, z) * z) / r
    let dotDzZ = 0
    for (let i = 0; i < dim; i++) {
      dotDzZ += dLdz[i] * z[i]
    }
    const invR = 1.0 / r
    for (let i = 0; i < dim; i++) {
      dyBuf[i] = (dLdz[i] - dotDzZ * z[i]) * invR
    }

    // dL/dB += scale * outer(dy, u)
    for (let i = 0; i < dim; i++) {
      const sDy = scale * dyBuf[i]
      const bOff = i * rank
      for (let k = 0; k < rank; k++) {
        gradB[bOff + k] += sDy * u[k]
      }
    }

    // du = scale * B^T * dy
    for (let k = 0; k < rank; k++) {
      let sum = 0
      for (let i = 0; i < dim; i++) {
        sum += B[i * rank + k] * dyBuf[i]
      }
      duBuf[k] = scale * sum
    }

    // dL/dA += outer(du, x)
    for (let k = 0; k < rank; k++) {
      const duk = duBuf[k]
      const aOff = k * dim
      for (let j = 0; j < dim; j++) {
        gradA[aOff + j] += duk * x[j]
      }
    }
  }

  const dLdzA = new Float32Array(dim)
  const dLdzP = new Float32Array(dim)
  const dLdzN = new Float32Array(dim)

  for (let p = 0; p < pairs.length; p++) {
    const pair = pairs[p]
    const xA = encodeBase(pair.anchor, dim)
    const xP = encodeBase(pair.positive, dim)
    const xN = encodeBase(pair.negative, dim)

    const fA = forwardWithLora(xA, baseW, adapter)
    const fP = forwardWithLora(xP, baseW, adapter)
    const fN = forwardWithLora(xN, baseW, adapter)

    let sPos = 0
    let sNeg = 0
    for (let i = 0; i < dim; i++) {
      sPos += fA.z[i] * fP.z[i]
      sNeg += fA.z[i] * fN.z[i]
    }
    totalPosSim += sPos
    totalNegSim += sNeg

    // Numerically stable softplus: log(1 + exp(delta))
    const delta = (sNeg - sPos) / tau
    const loss =
      delta > 20
        ? delta
        : delta < -20
          ? Math.exp(delta)
          : Math.log1p(Math.exp(delta))
    totalLoss += loss

    // sigmoid(delta) / tau
    const sig = 1.0 / (1.0 + Math.exp(-Math.max(-40, Math.min(40, delta))))
    const w = sig / tau

    for (let i = 0; i < dim; i++) {
      dLdzA[i] = w * (fN.z[i] - fP.z[i])
      dLdzP[i] = -w * fA.z[i]
      dLdzN[i] = w * fA.z[i]
    }

    accumulateSampleGrad(dLdzA, fA.z, fA.r, fA.u, xA)
    accumulateSampleGrad(dLdzP, fP.z, fP.r, fP.u, xP)
    accumulateSampleGrad(dLdzN, fN.z, fN.r, fN.u, xN)
  }

  const invBatch = 1.0 / pairs.length
  let gradSqNorm = 0
  for (let i = 0; i < gradA.length; i++) {
    gradA[i] *= invBatch
    gradSqNorm += gradA[i] * gradA[i]
  }
  for (let i = 0; i < gradB.length; i++) {
    gradB[i] *= invBatch
    gradSqNorm += gradB[i] * gradB[i]
  }

  const gradNorm = Math.sqrt(gradSqNorm)
  const maxGradNorm = 5.0
  const clipScale = gradNorm > maxGradNorm ? maxGradNorm / gradNorm : 1.0
  const stepLr = lr * clipScale

  for (let i = 0; i < A.length; i++) {
    A[i] -= stepLr * gradA[i]
  }
  for (let i = 0; i < B.length; i++) {
    B[i] -= stepLr * gradB[i]
  }

  const meanLoss = totalLoss * invBatch
  const meanPosSim = totalPosSim * invBatch
  const meanNegSim = totalNegSim * invBatch

  adapter.step += 1
  adapter.lossHistory.push(meanLoss)
  adapter.sha256 = computeAdapterDigest(A, B, adapter.taskId, rank, alpha)
  adapter.updatedAt = new Date().toISOString()

  return {
    step: adapter.step,
    loss: meanLoss,
    posSim: meanPosSim,
    negSim: meanNegSim,
    gradNorm,
    adapterSha256: adapter.sha256,
  }
}

/**
 * Pure Web `ReadableStream<TrainStepEvent>` emitting live contrastive fine-tuning
 * metrics on every step so UIs can plot a real decreasing loss curve.
 */
export function createLoraTrainStream(
  opts: LoraTrainStreamOptions
): ReadableStream<TrainStepEvent> {
  const totalSteps = opts.steps ?? 20
  const lr = opts.lr ?? 0.08
  const baseW = opts.baseW ?? createIdentityMatrix(opts.adapter.dim)
  const temperature = opts.temperature ?? 0.25

  if (!Number.isInteger(totalSteps) || totalSteps < 1) {
    throw new RangeError(`steps must be a positive integer, got ${totalSteps}`)
  }

  let currentStep = 0
  const startMs = typeof performance !== 'undefined' ? performance.now() : Date.now()
  const nowMs = (): number =>
    typeof performance !== 'undefined' ? performance.now() : Date.now()

  return new ReadableStream<TrainStepEvent>({
    pull(controller) {
      if (currentStep >= totalSteps) {
        controller.close()
        return
      }
      currentStep += 1
      const res = trainContrastiveStep(opts.adapter, opts.pairs, lr, {
        baseW,
        temperature,
      })
      const elapsedMs = Number((nowMs() - startMs).toFixed(2))
      controller.enqueue({
        step: currentStep,
        totalSteps,
        loss: res.loss,
        posSim: res.posSim,
        negSim: res.negSim,
        lr,
        elapsedMs,
        adapterSha256: res.adapterSha256,
      })
    },
  })
}

/**
 * Computes exact `W' = W + (alpha / rank) * (B * A)` in Float32 and returns
 * the merged weight tensor together with its SHA-256 CAS digest and byte size.
 */
export function mergeLoraIntoBase(
  baseW: Float32Array,
  adapter: LoraAdapter
): MergeLoraResult {
  const { dim, rank, alpha, A, B } = adapter
  const expectedLen = dim * dim
  if (baseW.length !== expectedLen) {
    throw new RangeError(
      `baseW length mismatch: expected ${expectedLen} (${dim}x${dim}), got ${baseW.length}`
    )
  }

  const scale = alpha / rank
  const mergedWeights = new Float32Array(expectedLen)

  for (let i = 0; i < dim; i++) {
    const wRowOff = i * dim
    const bRowOff = i * rank
    for (let j = 0; j < dim; j++) {
      let ba = 0
      for (let k = 0; k < rank; k++) {
        ba += B[bRowOff + k] * A[k * dim + j]
      }
      mergedWeights[wRowOff + j] = baseW[wRowOff + j] + scale * ba
    }
  }

  return {
    mergedWeights,
    sha256: computeFloat32Sha256(mergedWeights),
    bytes: mergedWeights.byteLength,
    dim,
  }
}

/**
 * Estimates realistic VRAM footprint (MB) for Unsloth-style QLoRA / LoRA fine-tuning
 * across `nf4`, `q8`, and `fp16` base quantizations, plus percentage VRAM savings vs `fp16`.
 */
export function estimateUnslothMemory(
  paramsBillion: number,
  rank: number,
  quant: QuantizationKind,
  batchSize = 4
): UnslothMemoryEstimate {
  if (!(paramsBillion > 0) || !Number.isFinite(paramsBillion)) {
    throw new RangeError(`paramsBillion must be > 0, got ${paramsBillion}`)
  }
  if (!Number.isInteger(rank) || rank < 1) {
    throw new RangeError(`rank must be a positive integer, got ${rank}`)
  }
  if (!Number.isInteger(batchSize) || batchSize < 1) {
    throw new RangeError(`batchSize must be a positive integer, got ${batchSize}`)
  }

  const MB = 1024 * 1024
  const totalParams = paramsBillion * 1e9

  const computeForQuant = (
    q: QuantizationKind
  ): { baseMb: number; loraMb: number; optimizerMb: number; totalMb: number } => {
    // Bytes per base parameter (including double-quantization metadata for NF4 / scales for Q8)
    const bytesPerParam = q === 'nf4' ? 0.5625 : q === 'q8' ? 1.05 : 2.0
    const baseMb = (totalParams * bytesPerParam) / MB

    // Trainable LoRA adapter params (all attention + MLP projections: ~0.008 * rank/16 of base)
    const loraParams = totalParams * (0.008 * (rank / 16))
    // Weights + gradients in bf16/fp16 (4 bytes total per trainable param)
    const loraMb = (loraParams * 4) / MB

    // Optimizer states (8-bit paged AdamW for NF4/Q8: 2 bytes/param; 32-bit AdamW for FP16: 8 bytes/param)
    // plus activation / KV cache overhead (Unsloth smart gradient checkpointing reduces NF4/Q8 activations)
    const optBytesPerLoraParam = q === 'fp16' ? 8 : 2
    const activationMbPerBatch =
      paramsBillion * (q === 'nf4' ? 12 : q === 'q8' ? 18 : 36)
    const optimizerMb =
      (loraParams * optBytesPerLoraParam) / MB + batchSize * activationMbPerBatch

    const totalMb = baseMb + loraMb + optimizerMb
    return { baseMb, loraMb, optimizerMb, totalMb }
  }

  const fp16 = computeForQuant('fp16')
  const cur = quant === 'fp16' ? fp16 : computeForQuant(quant)
  const savingsPctVsFp16 =
    quant === 'fp16'
      ? 0
      : Number((((fp16.totalMb - cur.totalMb) / fp16.totalMb) * 100).toFixed(2))

  return {
    paramsBillion,
    rank,
    quant,
    batchSize,
    baseMb: Number(cur.baseMb.toFixed(2)),
    loraMb: Number(cur.loraMb.toFixed(2)),
    optimizerMb: Number(cur.optimizerMb.toFixed(2)),
    totalMb: Number(cur.totalMb.toFixed(2)),
    savingsPctVsFp16,
  }
}

// ---------------------------------------------------------------------------
// High-Level Sovereign Studio Engine
// ---------------------------------------------------------------------------

export class LoraStudioEngine {
  readonly dim: number
  readonly defaultRank: number
  readonly defaultAlpha: number
  baseWeights: Float32Array
  baseSha256: string
  readonly adapters: Map<TaskAdapterId, LoraAdapter>

  constructor(opts?: LoraStudioEngineOptions) {
    this.dim = opts?.dim ?? 64
    this.defaultRank = opts?.defaultRank ?? 8
    this.defaultAlpha = opts?.defaultAlpha ?? this.defaultRank * 2

    if (opts?.baseWeights) {
      if (opts.baseWeights.length !== this.dim * this.dim) {
        throw new RangeError(
          `baseWeights must have length ${this.dim * this.dim}, got ${opts.baseWeights.length}`
        )
      }
      this.baseWeights = new Float32Array(opts.baseWeights)
    } else {
      this.baseWeights = createIdentityMatrix(this.dim)
    }

    this.baseSha256 = computeFloat32Sha256(this.baseWeights)
    this.adapters = new Map<TaskAdapterId, LoraAdapter>()

    // Pre-register the 5 Jina-v3 / ModernBERT / lumiBERT task adapters
    for (const taskId of TASK_ADAPTER_IDS) {
      this.adapters.set(
        taskId,
        createLoraAdapter({
          taskId,
          dim: this.dim,
          rank: this.defaultRank,
          alpha: this.defaultAlpha,
          strategy: 'direct-task-finetune',
        })
      )
    }
  }

  getAdapter(taskId: TaskAdapterId): LoraAdapter {
    const adapter = this.adapters.get(taskId)
    if (!adapter) {
      throw new RangeError(`Unknown task adapter: ${taskId}`)
    }
    return adapter
  }

  listAdapters(): readonly LoraAdapterSummary[] {
    const out: LoraAdapterSummary[] = []
    for (const adapter of this.adapters.values()) {
      const paramCount = adapter.A.length + adapter.B.length
      out.push({
        id: adapter.id,
        taskId: adapter.taskId,
        strategy: adapter.strategy,
        dim: adapter.dim,
        rank: adapter.rank,
        alpha: adapter.alpha,
        sha256: adapter.sha256,
        step: adapter.step,
        lastLoss:
          adapter.lossHistory.length > 0
            ? adapter.lossHistory[adapter.lossHistory.length - 1]
            : null,
        lossHistory: [...adapter.lossHistory],
        paramCount,
        bytes: adapter.A.byteLength + adapter.B.byteLength,
      })
    }
    return out
  }

  createOrResetAdapter(
    taskId: TaskAdapterId,
    opts?: {
      readonly rank?: number
      readonly alpha?: number
      readonly strategy?: LoraStrategy
      readonly seedSalt?: string
    }
  ): LoraAdapter {
    const adapter = createLoraAdapter({
      taskId,
      dim: this.dim,
      rank: opts?.rank ?? this.defaultRank,
      alpha: opts?.alpha ?? (opts?.rank ?? this.defaultRank) * 2,
      strategy: opts?.strategy ?? 'direct-task-finetune',
      seedSalt: opts?.seedSalt,
    })
    this.adapters.set(taskId, adapter)
    return adapter
  }

  /**
   * Computes a normalized Float32 embedding using the current base weights,
   * an optional task LoRA adapter, and optional Matryoshka dimension truncation.
   */
  embed(
    text: string,
    opts?: {
      readonly taskId?: TaskAdapterId
      readonly matryoshkaDim?: number
    }
  ): Float32Array {
    const baseVec = encodeBase(text, this.dim)
    let projected: Float32Array
    if (opts?.taskId) {
      const adapter = this.getAdapter(opts.taskId)
      projected = forwardWithLora(baseVec, this.baseWeights, adapter).z
    } else {
      projected = normalizeL2(applyLinearMatrix(this.baseWeights, baseVec, this.dim))
    }

    if (opts?.matryoshkaDim !== undefined) {
      return sliceMatryoshka(projected, opts.matryoshkaDim)
    }
    return projected
  }

  similarity(
    textA: string,
    textB: string,
    opts?: {
      readonly taskId?: TaskAdapterId
      readonly matryoshkaDim?: number
    }
  ): number {
    const vecA = this.embed(textA, opts)
    const vecB = this.embed(textB, opts)
    return cosineSimilarity(vecA, vecB)
  }

  trainStep(
    taskId: TaskAdapterId,
    pairs: readonly ContrastivePair[],
    lr = 0.08,
    temperature = 0.25
  ): TrainStepResult {
    const adapter = this.getAdapter(taskId)
    return trainContrastiveStep(adapter, pairs, lr, {
      baseW: this.baseWeights,
      temperature,
    })
  }

  trainStream(
    taskId: TaskAdapterId,
    pairs: readonly ContrastivePair[],
    opts?: {
      readonly steps?: number
      readonly lr?: number
      readonly temperature?: number
    }
  ): ReadableStream<TrainStepEvent> {
    const adapter = this.getAdapter(taskId)
    return createLoraTrainStream({
      adapter,
      pairs,
      steps: opts?.steps,
      lr: opts?.lr,
      baseW: this.baseWeights,
      temperature: opts?.temperature,
    })
  }

  trainLivenessStream(
    taskId: TaskAdapterId,
    pairs: readonly ContrastivePair[],
    opts?: {
      readonly steps?: number
      readonly lr?: number
      readonly temperature?: number
    }
  ): ReadableStream<LivenessFrame<TrainStepEvent>> {
    const total = opts?.steps ?? 20
    return this.trainStream(taskId, pairs, opts).pipeThrough(
      createLivenessTransform<TrainStepEvent>({
        label: `lora-train:${taskId}`,
        total,
      })
    )
  }

  /**
   * Executes one of the two Jina-v3 LoRA adaptation strategies:
   *  - `'merge-then-stack'`: merges `W' = W + (α/r)·(B·A)` into a new base weight snapshot
   *    and initializes a fresh domain LoRA `(A2, B2)` on top.
   *  - `'direct-task-finetune'`: keeps `baseWeights` frozen and mutates the target `taskId`
   *    adapter directly with zero extra inference latency.
   */
  applyStrategy(
    taskId: TaskAdapterId,
    strategy: LoraStrategy,
    opts?: {
      readonly newRank?: number
      readonly newAlpha?: number
    }
  ): StrategyApplyResult {
    const current = this.getAdapter(taskId)

    if (strategy === 'merge-then-stack') {
      const merged = mergeLoraIntoBase(this.baseWeights, current)
      this.baseWeights = merged.mergedWeights
      this.baseSha256 = merged.sha256

      const stacked = createLoraAdapter({
        taskId,
        dim: this.dim,
        rank: opts?.newRank ?? current.rank,
        alpha: opts?.newAlpha ?? current.alpha,
        strategy: 'merge-then-stack',
        seedSalt: `stack-${merged.sha256.slice(0, 12)}`,
      })
      this.adapters.set(taskId, stacked)

      return {
        strategy,
        taskId,
        baseSha256: this.baseSha256,
        adapter: stacked,
        merged,
        extraInferenceLatency: true,
      }
    }

    current.strategy = 'direct-task-finetune'
    current.updatedAt = new Date().toISOString()
    return {
      strategy: 'direct-task-finetune',
      taskId,
      baseSha256: this.baseSha256,
      adapter: current,
      merged: null,
      extraInferenceLatency: false,
    }
  }

  mergeAdapter(taskId: TaskAdapterId): MergeLoraResult {
    const adapter = this.getAdapter(taskId)
    return mergeLoraIntoBase(this.baseWeights, adapter)
  }

  estimateMemory(
    paramsBillion: number,
    rank: number,
    quant: QuantizationKind,
    batchSize = 4
  ): UnslothMemoryEstimate {
    return estimateUnslothMemory(paramsBillion, rank, quant, batchSize)
  }
}
