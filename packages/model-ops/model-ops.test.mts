import assert from 'node:assert/strict'
import test from 'node:test'
import {
  CURATED_MODEL_SPECS,
  getCuratedModelSpec,
  stampModelBlueprint,
} from './index.mts'

test('b8g-model-ops: Inkling 2 Mini preset expands 24 physical -> 48 logical layers', () => {
  const spec = getCuratedModelSpec('inkling-2-mini')
  assert.ok(spec, 'inkling-2-mini must exist')
  const bp = stampModelBlueprint(spec)
  assert.equal(bp.physicalLayers, 24)
  assert.equal(bp.logicalLayers, 48)
  assert.equal(bp.loopFactor, 2)
  assert.equal(bp.verifiedReference?.totalParamsB, 26.4)
  assert.equal(bp.verifiedReference?.effectiveActiveParamsB, 5.4)
  assert.equal(bp.arenaSlabs.every((s) => s.aligned64), true)
})

test('b8g-model-ops: B8G Sovereign MoE v1 is boundary-verified with 16 physical -> 32 logical layers', () => {
  const spec = getCuratedModelSpec('b8g-sovereign-moe-v1')
  assert.ok(spec, 'b8g-sovereign-moe-v1 must exist')
  const bp = stampModelBlueprint(spec)
  assert.equal(bp.physicalLayers, 16)
  assert.equal(bp.logicalLayers, 32)
  assert.equal(bp.boundaryVerified, true)
  assert.ok(bp.estimatedEffectiveActiveParams > bp.estimatedBaseActiveParams)
  assert.ok(bp.estimatedTotalParams > bp.estimatedBaseActiveParams)
})

test('b8g-model-ops: LittleBit 0.25 BPW preset shrinks memory drastically vs FP16', () => {
  const edge = getCuratedModelSpec('b8g-edge-littlebit')
  assert.ok(edge)
  const bpSub1 = stampModelBlueprint(edge)
  const bpFp16 = stampModelBlueprint({ ...edge, quant: 'fp16' })
  assert.ok(bpSub1.estimatedWeightBytes * 50 < bpFp16.estimatedWeightBytes, '0.25 BPW must be >50x smaller than FP16')
})

test('b8g-model-ops: all curated presets stamp cleanly with 64-byte aligned B8G_ARENA_V1 slabs', () => {
  assert.ok(CURATED_MODEL_SPECS.length >= 4)
  for (const spec of CURATED_MODEL_SPECS) {
    const bp = stampModelBlueprint(spec)
    assert.ok(bp.totalArenaBytes > 0)
    assert.ok(bp.arenaSlabs.length === 5)
    assert.equal(bp.arenaSlabs.every((s) => s.aligned64), true)
  }
})
