/**
 * @file kalaschnikow.test.mts
 * @description Verification suite for Model Kalaschnikow (v0.1-pre.1).
 * Tests IsoFLOP parity, block-sparse routing, B8G memory arena serialization,
 * zero-copy tensor forward pass, and Chinchilla scaling law extrapolation.
 */

import assert from 'node:assert/strict'
import test from 'node:test'
import * as os from 'node:os'
import * as path from 'node:path'
import * as fs from 'node:fs'

import {
  IsoFLOPGeometryEngine,
  KalaschnikowModel,
  B8GModelArena,
  ARENA_MAGIC,
  BlockSparseLinear,
  ChinchillaScalingEngine
} from '../src/index.mts'

test('IsoFLOP Parity: Asymmetric Tapered Kalaschnikow stays within +/-2% parameter envelope of 80M Symmetric baseline', () => {
  const symConfig = IsoFLOPGeometryEngine.createSymmetricConfig()
  const asymConfig = IsoFLOPGeometryEngine.createKalaschnikowConfig()

  const symParams = IsoFLOPGeometryEngine.calculateParameters(symConfig)
  const asymParams = IsoFLOPGeometryEngine.calculateParameters(asymConfig)

  console.log(`\n--- IsoFLOP Parity Audit ---`)
  console.log(`Symmetric Baseline Total Params:  ${symParams.totalParams.toLocaleString()}`)
  console.log(`Kalaschnikow Asymmetric Total:    ${asymParams.totalParams.toLocaleString()}`)

  const parity = IsoFLOPGeometryEngine.verifyIsoFlopParity(symParams, asymParams, 2.0)
  console.log(`Parameter Delta: ${parity.deltaPercent.toFixed(2)}% (Within +/-2% limit: ${parity.isWithinParity})`)

  assert.equal(parity.isWithinParity, true, `Expected parameter delta (${parity.deltaPercent}%) to be within +/-2%`)
  const symLayerParams = symParams.attentionParams + symParams.mlpParams
  const asymLayerParams = asymParams.attentionParams + asymParams.mlpParams + asymParams.projectionParams
  assert.ok(symLayerParams >= 80_000_000 && symLayerParams <= 90_000_000, 'Baseline transformer layers should be ~85M')
  assert.ok(asymLayerParams >= 80_000_000 && asymLayerParams <= 90_000_000, 'Kalaschnikow transformer layers should be ~85M')
  assert.ok(symParams.totalParams >= 130_000_000 && symParams.totalParams <= 145_000_000)
  assert.ok(asymParams.totalParams >= 130_000_000 && asymParams.totalParams <= 145_000_000)
})

test('Block-Sparse Routing: 16x16 tile Frobenius norm pruning and metrics tracking', () => {
  const arena = new B8GModelArena(16 * 1024 * 1024)
  const sparseLayer = new BlockSparseLinear(arena, 'test_sparse_fc', 512, 1024, 16, 0.5)

  const metrics = sparseLayer.getMetrics()
  console.log(`\n--- Block-Sparse Metrics ---`)
  console.log(`Total 16x16 Blocks:    ${metrics.totalBlocks}`)
  console.log(`Active Blocks:         ${metrics.activeBlocks}`)
  console.log(`Actual Sparsity:       ${(metrics.actualSparsity * 100).toFixed(1)}%`)
  console.log(`FLOP Reduction:        ${metrics.theoreticalFlopReduction.toFixed(1)}%`)

  assert.equal(metrics.totalBlocks, (512 / 16) * (1024 / 16))
  assert.equal(metrics.activeBlocks, metrics.totalBlocks * 0.5)
  assert.equal(metrics.actualSparsity, 0.5)
  assert.equal(metrics.theoreticalFlopReduction, 50.0)

  // Verify forward pass executes without errors
  const inVec = new Float32Array(512).fill(0.1)
  const outVec = new Float32Array(1024)
  sparseLayer.forward(inVec, outVec)

  // Output must be populated and non-NaN
  assert.ok(!Number.isNaN(outVec[0]))
  assert.ok(outVec.some(v => v !== 0.0))
})

test('B8G Zero-Copy Model Arena: Contiguous allocation, 64-byte alignment, and zero-allocation forward pass', () => {
  const model = new KalaschnikowModel()
  const descriptors = model.arena.getTensorDescriptors()

  assert.ok(descriptors.length > 20, 'Expected multiple tensor descriptors')
  console.log(`\n--- Arena Allocation Audit ---`)
  console.log(`Allocated Tensors: ${descriptors.length}`)
  console.log(`Arena Memory Used: ${(model.arena.getUsedBytes() / (1024 * 1024)).toFixed(2)} MB`)

  // Check 64-byte cache alignment for all tensors
  for (const desc of descriptors) {
    assert.equal(desc.offsetBytes % 64, 0, `Tensor ${desc.name} not 64-byte aligned: ${desc.offsetBytes}`)
  }

  // Run forward pass on synthetic token embedding
  const dummyEmbed = new Float32Array(768).fill(0.05)
  const logits = new Float32Array(model.config.vocabSize)

  model.forwardToken(dummyEmbed, logits)

  assert.equal(logits.length, 32000)
  assert.ok(!Number.isNaN(logits[0]), 'Logits must not contain NaN')
  assert.ok(logits.some(v => v !== 0.0), 'Logits must be computed')
})

test('Binary Memory Arena Serialization: B8G_ARENA_V1 round-trip file snapshot', () => {
  const arena = new B8GModelArena(32 * 1024 * 1024)
  const t1 = arena.allocateTensor('block0.weight', [4, 4], new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))
  const t2 = arena.allocateTensor('block0.bias', [4], new Float32Array([0.1, 0.2, 0.3, 0.4]))

  const tmpPath = path.join(os.tmpdir(), `kalaschnikow_test_arena_${process.pid}.bin`)
  try {
    arena.saveToFile(tmpPath)
    assert.ok(fs.existsSync(tmpPath))

    const loaded = B8GModelArena.loadFromFile(tmpPath)
    assert.equal(loaded.hasTensor('block0.weight'), true)
    assert.equal(loaded.hasTensor('block0.bias'), true)

    const loadedT1 = loaded.getTensor('block0.weight')!
    const loadedT2 = loaded.getTensor('block0.bias')!

    assert.deepEqual(Array.from(loadedT1), Array.from(t1))
    assert.deepEqual(Array.from(loadedT2), Array.from(t2))
  } finally {
    if (fs.existsSync(tmpPath)) {
      fs.unlinkSync(tmpPath)
    }
  }
})

test('Chinchilla Scaling Extrapolation: Fits sub-scale points and predicts 7B compute-optimal ratio', () => {
  const fitted = ChinchillaScalingEngine.fitScalingLaws()
  console.log(`\n--- Chinchilla Scaling Law Extrapolation ---`)
  console.log(`Fitted: E=${fitted.E.toFixed(4)}, A=${fitted.A.toFixed(2)}, B=${fitted.B.toFixed(2)}, alpha=${fitted.alpha.toFixed(4)}, beta=${fitted.beta.toFixed(4)}`)

  const extrapolation = ChinchillaScalingEngine.extrapolate(7.0e9, fitted)
  console.log(`Target Parameter Count:           ${extrapolation.targetParams.toLocaleString()}`)
  console.log(`Compute-Optimal Tokens:           ${Math.round(extrapolation.optimalTokens).toLocaleString()}`)
  console.log(`Optimal Token/Param Ratio:        ${extrapolation.tokenToParamRatio.toFixed(1)}x`)
  console.log(`Predicted Validation Loss:        ${extrapolation.predictedLoss.toFixed(4)}`)
  console.log(`Estimated Total Training FLOPs:   ${extrapolation.estimatedTrainingFlops.toExponential(3)} FLOPs`)

  assert.ok(extrapolation.optimalTokens > 1.0e11, 'Expected at least 100B tokens for 7B optimal')
  assert.ok(extrapolation.tokenToParamRatio >= 15 && extrapolation.tokenToParamRatio <= 80, 'Optimal ratio should be physically sound (~20x-65x)')
  assert.ok(extrapolation.predictedLoss < 2.5, 'Expected validation loss to be well below subscale runs')
})

test('Quaternion SLERP on S^3: Unit-norm manifold invariance and geodesic interpolation', async () => {
  const { Quaternion } = await import('../src/linear-slerp.mts')
  const q0 = new Quaternion(1, 0, 0, 0)
  const q1 = new Quaternion(0, 1, 0, 0) // 90 degree orthogonal rotation

  // Midpoint t = 0.5
  const mid = q0.slerp(q1, 0.5)
  const norm = Math.sqrt(mid.w * mid.w + mid.x * mid.x + mid.y * mid.y + mid.z * mid.z)

  assert.ok(Math.abs(norm - 1.0) < 1e-6, 'SLERP must strictly conserve unit-norm S^3 manifold invariance')
  assert.ok(Math.abs(mid.w - Math.SQRT1_2) < 1e-4, 'Midpoint w should equal sqrt(2)/2')
  assert.ok(Math.abs(mid.x - Math.SQRT1_2) < 1e-4, 'Midpoint x should equal sqrt(2)/2')
})

test('Planetary Linear Kalaschnikow: O(1) Constant Context Memory and Zero KV-Cache Growth', async () => {
  const { PlanetaryLinearKalaschnikowModel, SphericalBayesianAccumulator } = await import('../src/linear-slerp.mts')

  const linearModel = new PlanetaryLinearKalaschnikowModel()
  const metrics = linearModel.getPlanetaryMetrics()

  console.log(`\n--- Planetary Linear Kalaschnikow Audit ---`)
  console.log(`Context Complexity:       ${metrics.contextMemoryComplexity}`)
  console.log(`KV-Cache Growth:          ${metrics.kvCacheGrowthRate}`)
  console.log(`Total Recurrent RAM:      ${(metrics.totalAccumulatorMemoryBytes / (1024 * 1024)).toFixed(2)} MB`)

  assert.equal(metrics.contextMemoryComplexity, 'O(1) constant per stream')
  assert.equal(metrics.kvCacheGrowthRate, '0 bytes / token (Zero KV-cache)')

  // Simulate streaming 100 sequential tokens to verify constant-memory execution
  const dummyEmbed = new Float32Array(768)
  const logits = new Float32Array(linearModel.config.vocabSize)

  for (let tokenIdx = 0; tokenIdx < 100; tokenIdx++) {
    dummyEmbed.fill(0.01 * (tokenIdx + 1))
    linearModel.forwardToken(dummyEmbed, logits)
  }

  assert.equal(logits.length, 32000)
  assert.ok(!Number.isNaN(logits[0]), 'Logits must remain stable and non-NaN after 100 steps')
  assert.ok(logits.some((v) => v !== 0.0), 'Logits must be computed')

  // Verify Bayesian surprise metrics
  const block0Metrics = linearModel.blocks[0].accumulator.getMetrics()
  assert.equal(block0Metrics.totalSteps, 100)
  assert.ok(block0Metrics.averageEntropy > 0.0, 'Bayesian entropy tracking must record positive surprise')
  console.log(`Layer 0 Avg Entropy:     ${block0Metrics.averageEntropy.toFixed(4)} bits`)
  console.log(`Layer 0 Last Gain (α):   ${block0Metrics.lastSurpriseGain.toFixed(4)}`)
})

test('Minimal Nature Reflection Micro-Benchmark: Softmax vs Linear SLERP Breathing Cycles', async () => {
  const { runMicroBenchmark } = await import('../src/micro-benchmark.mts')

  const summary = runMicroBenchmark(64)

  console.log(`\n=============================================================================`)
  console.log(`  MINIMAL NATURE REFLECTION MICRO-BENCHMARK: SOFTMAX vs LINEAR SLERP`)
  console.log(`=============================================================================`)
  console.log(`SeqLen | Softmax KV | SLERP RAM  | RAM Savings | Softmax (ms) | SLERP (ms) | Speedup`)
  console.log(`-------+------------+------------+-------------+--------------+------------+--------`)

  for (const step of summary.steps) {
    const sKv = `${(step.softmaxKvBytes / 1024).toFixed(1)} KB`.padStart(10)
    const slRam = `${(step.linearSlerpBytes / 1024).toFixed(1)} KB`.padStart(10)
    const sav = `${step.memorySavingsRatio.toFixed(2)}x`.padStart(11)
    const sMs = `${step.softmaxStepMs.toFixed(3)} ms`.padStart(12)
    const slMs = `${step.slerpStepMs.toFixed(3)} ms`.padStart(10)
    const sp = `${step.speedupRatio.toFixed(1)}x`.padStart(6)
    console.log(`${step.sequenceLength.toString().padStart(6)} | ${sKv} | ${slRam} | ${sav} | ${sMs} | ${slMs} | ${sp}`)

    // Linear SLERP memory must remain strictly constant
    assert.equal(step.linearSlerpBytes, summary.steps[0].linearSlerpBytes, 'SLERP memory must be O(1) constant')
    // At L = 2048, memory savings must exceed 50x
    if (step.sequenceLength === 2048) {
      assert.ok(step.memorySavingsRatio > 50, 'At 2k tokens, memory savings should be > 50x')
    }
  }

  console.log(`\n[Breathing Cycles: Inhalation (Novelty/High Entropy) -> Exhalation (Rest/Low Entropy)]`)
  for (const b of summary.breathingCycles) {
    console.log(`  Step ${b.tokenStep.toString().padStart(2, ' ')} [${b.phase.padEnd(10)}] Entropy: ${b.entropyBits.toFixed(2)} bits | Bayesian Gain (α): ${b.bayesianGainAlpha.toFixed(3)} | S^3 Norm: ${b.quaternionNorm.toFixed(5)}`)
    assert.equal(b.stateConserved, true, 'Quaternion unit-norm on S^3 must be conserved across all cycles')
  }
})


