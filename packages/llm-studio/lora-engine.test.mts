/**
 * @file intersections/b8g-llm-studio/lora-engine.test.mts
 * @description Unit tests for the Zero-Dep Sovereign LoRA / Unsloth-Style Embedding & Fine-Tune Engine.
 */

import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import {
  TASK_ADAPTER_IDS,
  LoraStudioEngine,
  encodeBase,
  sliceMatryoshka,
  cosineSimilarity,
  l2Norm,
  createIdentityMatrix,
  createLoraAdapter,
  trainContrastiveStep,
  createLoraTrainStream,
  mergeLoraIntoBase,
  estimateUnslothMemory,
  applyLinearMatrix,
  normalizeL2,
  type ContrastivePair,
  type TrainStepEvent,
} from './lora-engine.mts'

const SAMPLE_PAIRS: readonly ContrastivePair[] = [
  {
    anchor: 'fix typescript erasableSyntaxOnly enum error',
    positive: 'replace enum with const union type for erasableSyntaxOnly compliance',
    negative: 'baked sourdough bread fermentation temperature schedule',
  },
  {
    anchor: 'zero copy sharedarraybuffer microkernel arena',
    positive: 'lock free ring buffer over SharedArrayBuffer memory arena',
    negative: 'alpine ski lift pass winter resort pricing',
  },
  {
    anchor: 'matryoshka embedding truncation cosine similarity',
    positive: 'slice vector prefix dimensions and L2 renormalize for retrieval',
    negative: 'classical baroque harpsichord tuning temperament',
  },
]

describe('LoraStudioEngine & Float32 Math', () => {
  it('encodeBase is deterministic, unit-norm, and preserves semantic overlap', () => {
    const a1 = encodeBase('sovereign microkernel wasm runtime', 64)
    const a2 = encodeBase('sovereign microkernel wasm runtime', 64)
    const similar = encodeBase('sovereign microkernel runtime execution', 64)
    const unrelated = encodeBase('chocolate cake strawberry frosting recipe', 64)

    assert.equal(a1.length, 64)
    assert.deepEqual(a1, a2)
    assert.ok(Math.abs(l2Norm(a1) - 1.0) < 1e-5, `expected unit norm, got ${l2Norm(a1)}`)
    assert.ok(Math.abs(cosineSimilarity(a1, a2) - 1.0) < 1e-5)

    const simRelated = cosineSimilarity(a1, similar)
    const simUnrelated = cosineSimilarity(a1, unrelated)
    assert.ok(
      simRelated > simUnrelated,
      `expected related (${simRelated}) > unrelated (${simUnrelated})`
    )
  })

  it('sliceMatryoshka preserves unit norm across 64 -> 32 -> 16 -> 8 dims', () => {
    const base = encodeBase('jina v3 matryoshka representation learning', 64)
    for (const targetDim of [64, 32, 16, 8]) {
      const sliced = sliceMatryoshka(base, targetDim)
      assert.equal(sliced.length, targetDim)
      const norm = l2Norm(sliced)
      assert.ok(
        Math.abs(norm - 1.0) < 1e-5,
        `expected unit norm at dim=${targetDim}, got ${norm}`
      )
    }

    assert.throws(() => sliceMatryoshka(base, 0), RangeError)
    assert.throws(() => sliceMatryoshka(base, 65), RangeError)
  })

  it('trainContrastiveStep strictly decreases loss and increases anchor-positive similarity', () => {
    const adapter = createLoraAdapter({
      taskId: 'code-search',
      dim: 64,
      rank: 8,
      alpha: 16,
    })
    const initialSha = adapter.sha256
    const baseW = createIdentityMatrix(64)

    const first = trainContrastiveStep(adapter, SAMPLE_PAIRS, 0.1, { baseW })
    let last = first
    for (let i = 0; i < 24; i++) {
      last = trainContrastiveStep(adapter, SAMPLE_PAIRS, 0.1, { baseW })
    }

    assert.equal(adapter.step, 25)
    assert.equal(adapter.lossHistory.length, 25)
    assert.notEqual(adapter.sha256, initialSha)
    assert.ok(
      last.loss < first.loss * 0.5,
      `expected substantial loss decrease: first=${first.loss}, last=${last.loss}`
    )
    assert.ok(
      last.posSim > first.posSim,
      `expected posSim to increase: first=${first.posSim}, last=${last.posSim}`
    )
    assert.ok(
      last.posSim > last.negSim,
      `expected posSim (${last.posSim}) > negSim (${last.negSim})`
    )
  })

  it('createLoraTrainStream emits live Web ReadableStream events with decreasing loss', async () => {
    const adapter = createLoraAdapter({
      taskId: 'error-repair',
      dim: 64,
      rank: 8,
      alpha: 16,
    })
    const stream = createLoraTrainStream({
      adapter,
      pairs: SAMPLE_PAIRS,
      steps: 15,
      lr: 0.1,
    })

    const events: TrainStepEvent[] = []
    const reader = stream.getReader()
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      events.push(value)
    }

    assert.equal(events.length, 15)
    assert.equal(events[0].step, 1)
    assert.equal(events[0].totalSteps, 15)
    assert.equal(events[14].step, 15)
    assert.ok(events[14].elapsedMs >= 0)
    assert.ok(
      events[14].loss < events[0].loss,
      `expected final stream loss (${events[14].loss}) < initial (${events[0].loss})`
    )
  })

  it('mergeLoraIntoBase computes numerically exact W\' = W + (alpha/r) * B * A', () => {
    const dim = 16
    const rank = 4
    const alpha = 12
    const adapter = createLoraAdapter({
      taskId: 'retrieval.query',
      dim,
      rank,
      alpha,
    })

    // Populate B with non-zero deterministic values so B * A != 0
    for (let i = 0; i < adapter.B.length; i++) {
      adapter.B[i] = ((i % 7) - 3) * 0.125
    }

    const baseW = createIdentityMatrix(dim)
    baseW[0] = 1.5
    baseW[dim + 2] = -0.25

    const merged = mergeLoraIntoBase(baseW, adapter)
    assert.equal(merged.dim, dim)
    assert.equal(merged.bytes, dim * dim * 4)
    assert.equal(merged.sha256.length, 64)

    // Verify element-by-element exact formula W'[i,j] = W[i,j] + (alpha/r) * sum_k B[i,k] * A[k,j]
    const scale = alpha / rank
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        let expected = baseW[i * dim + j]
        for (let k = 0; k < rank; k++) {
          expected += scale * adapter.B[i * rank + k] * adapter.A[k * dim + j]
        }
        const actual = merged.mergedWeights[i * dim + j]
        assert.ok(
          Math.abs(actual - expected) < 1e-5,
          `mismatch at (${i},${j}): actual=${actual}, expected=${expected}`
        )
      }
    }

    // Verify merged linear projection matches unmerged W*x + (alpha/r)*B*(A*x)
    const engine = new LoraStudioEngine({ dim, defaultRank: rank, defaultAlpha: alpha, baseWeights: baseW })
    engine.adapters.set('retrieval.query', adapter)
    const x = encodeBase('verify merged weight equivalence', dim)
    const viaEngine = engine.embed('verify merged weight equivalence', { taskId: 'retrieval.query' })
    const viaMerged = normalizeL2(applyLinearMatrix(merged.mergedWeights, x, dim))

    assert.ok(
      cosineSimilarity(viaEngine, viaMerged) > 0.99999,
      `expected merged projection cosine ~1.0, got ${cosineSimilarity(viaEngine, viaMerged)}`
    )
  })

  it('supports both Jina-v3 strategies: direct-task-finetune and merge-then-stack', () => {
    const engine = new LoraStudioEngine({ dim: 32, defaultRank: 4, defaultAlpha: 8 })
    assert.equal(engine.listAdapters().length, TASK_ADAPTER_IDS.length)

    // 1. direct-task-finetune keeps baseW frozen and updates task adapter in-place
    const initialBaseSha = engine.baseSha256
    const directRes = engine.applyStrategy('retrieval.passage', 'direct-task-finetune')
    assert.equal(directRes.extraInferenceLatency, false)
    assert.equal(directRes.merged, null)
    assert.equal(engine.baseSha256, initialBaseSha)

    for (let i = 0; i < 8; i++) {
      engine.trainStep('retrieval.passage', SAMPLE_PAIRS, 0.1)
    }
    assert.equal(engine.baseSha256, initialBaseSha)

    // Embedding before merge-then-stack
    const beforeVec = engine.embed('sharedarraybuffer microkernel arena', {
      taskId: 'retrieval.passage',
    })

    // 2. merge-then-stack merges W' into base snapshot and initializes fresh zero-B LoRA on top
    const stackRes = engine.applyStrategy('retrieval.passage', 'merge-then-stack')
    assert.equal(stackRes.extraInferenceLatency, true)
    assert.ok(stackRes.merged !== null)
    assert.notEqual(engine.baseSha256, initialBaseSha)
    assert.equal(stackRes.adapter.step, 0)
    assert.equal(l2Norm(stackRes.adapter.B), 0)

    // Immediately after merge-then-stack (with B2 = 0), embeddings must be preserved!
    const afterVec = engine.embed('sharedarraybuffer microkernel arena', {
      taskId: 'retrieval.passage',
    })
    assert.ok(
      cosineSimilarity(beforeVec, afterVec) > 0.99999,
      `expected preserved embedding after merge-then-stack, got ${cosineSimilarity(beforeVec, afterVec)}`
    )
  })

  it('trainLivenessStream wraps training progress with pending -> progress -> done LivenessFrames', async () => {
    const engine = new LoraStudioEngine({ dim: 32, defaultRank: 4, defaultAlpha: 8 })
    const stream = engine.trainLivenessStream('text-matching', SAMPLE_PAIRS, {
      steps: 5,
      lr: 0.08,
    })

    const phases: string[] = []
    const reader = stream.getReader()
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      phases.push(value.phase)
    }

    assert.equal(phases[0], 'pending')
    assert.equal(phases[phases.length - 1], 'done')
    assert.equal(phases.filter((p) => p === 'progress').length, 5)
  })

  it('estimateUnslothMemory reports ~65-75% VRAM savings for NF4 QLoRA vs FP16', () => {
    const fp16 = estimateUnslothMemory(7, 16, 'fp16', 4)
    const q8 = estimateUnslothMemory(7, 16, 'q8', 4)
    const nf4 = estimateUnslothMemory(7, 16, 'nf4', 4)

    assert.equal(fp16.savingsPctVsFp16, 0)
    assert.ok(q8.totalMb < fp16.totalMb)
    assert.ok(nf4.totalMb < q8.totalMb)
    assert.ok(
      nf4.savingsPctVsFp16 >= 65 && nf4.savingsPctVsFp16 <= 75,
      `expected NF4 VRAM savings in [65%, 75%], got ${nf4.savingsPctVsFp16}% (nf4=${nf4.totalMb}MB vs fp16=${fp16.totalMb}MB)`
    )
    assert.ok(nf4.baseMb > 0 && nf4.loraMb > 0 && nf4.optimizerMb > 0)
  })
})
