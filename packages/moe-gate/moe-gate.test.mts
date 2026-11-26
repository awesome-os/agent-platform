import assert from 'node:assert/strict'
import test from 'node:test'
import { B8gVerifiedMoERouter } from './index.mts'

test('b8g-moe-gate: executes 3-pass looped MoE with Block Attention Residual cleanly', () => {
  const router = new B8gVerifiedMoERouter({
    hiddenSize: 4,
    topK: 2,
    loopPasses: 3,
    maxIngressL2Norm: 50,
    maxEgressL2Norm: 50,
    loopBlendAlpha: 0.4,
  })
  router.registerExpert({
    id: 1,
    name: 'math-expert',
    requiredScope: 'public',
    routerKey: new Float32Array([1, 0, 0, 0]),
    forward: (inp, out) => {
      for (let i = 0; i < inp.length; i += 1) out[i] = inp[i]! * 1.1
    },
  })
  router.registerExpert({
    id: 2,
    name: 'code-expert',
    requiredScope: 'public',
    routerKey: new Float32Array([0.8, 0.2, 0, 0]),
    forward: (inp, out) => {
      for (let i = 0; i < inp.length; i += 1) out[i] = inp[i]! * 0.9
    },
  })

  const res = router.dispatch(new Float32Array([1, 1, 0, 0]), {
    tenantId: 'frank',
    allowedScopes: ['public'],
  })
  assert.equal(res.ok, true)
  assert.equal(res.passesExecuted, 3)
  assert.equal(res.activeExpertsExecuted, 6) // 2 experts * 3 passes
  assert.equal(res.quarantinedCount, 0)
  assert.equal(res.output.every((v) => Number.isFinite(v)), true)
})

test('b8g-moe-gate: enforces CapabilityToken tenant isolation on private experts', () => {
  const router = new B8gVerifiedMoERouter({
    hiddenSize: 4,
    topK: 1,
    loopPasses: 1,
    maxIngressL2Norm: 50,
    maxEgressL2Norm: 50,
    loopBlendAlpha: 0.5,
  })
  let privateCalled = false
  router.registerExpert({
    id: 10,
    name: 'secret-tenant-expert',
    requiredScope: 'tenant:enterprise-a',
    routerKey: new Float32Array([100, 0, 0, 0]),
    forward: (_inp, out) => {
      privateCalled = true
      out.fill(9)
    },
  })
  router.registerExpert({
    id: 11,
    name: 'public-fallback',
    requiredScope: 'public',
    routerKey: new Float32Array([1, 0, 0, 0]),
    forward: (inp, out) => {
      out.set(inp)
    },
  })

  const res = router.dispatch(new Float32Array([1, 0, 0, 0]), {
    tenantId: 'tenant-b',
    allowedScopes: ['public'],
  })
  assert.equal(res.ok, true)
  assert.equal(privateCalled, false, 'Private expert must NEVER be invoked without matching scope')
  assert.equal(res.capabilityDenials, 1)
  assert.equal(res.activeExpertsExecuted, 1)
})

test('b8g-moe-gate: quarantines NaN-producing and energy-exploding experts without poisoning output', () => {
  const router = new B8gVerifiedMoERouter({
    hiddenSize: 4,
    topK: 3,
    loopPasses: 2,
    maxIngressL2Norm: 50,
    maxEgressL2Norm: 20,
    loopBlendAlpha: 0.5,
  })
  router.registerExpert({
    id: 1,
    name: 'healthy-expert',
    requiredScope: 'public',
    routerKey: new Float32Array([1, 0, 0, 0]),
    forward: (inp, out) => out.set(inp),
  })
  router.registerExpert({
    id: 2,
    name: 'nan-expert',
    requiredScope: 'public',
    routerKey: new Float32Array([1, 0, 0, 0]),
    forward: (_inp, out) => {
      out[1] = Number.NaN
    },
  })
  router.registerExpert({
    id: 3,
    name: 'exploding-expert',
    requiredScope: 'public',
    routerKey: new Float32Array([1, 0, 0, 0]),
    forward: (_inp, out) => out.fill(999),
  })

  const res = router.dispatch(new Float32Array([1, 0, 0, 0]), {
    tenantId: 'frank',
    allowedScopes: ['public'],
  })
  assert.equal(res.ok, true)
  assert.equal(res.quarantinedCount, 2)
  assert.equal(router.isQuarantined(2), true)
  assert.equal(router.isQuarantined(3), true)
  assert.equal(router.isQuarantined(1), false)
  assert.equal(res.output.every((v) => Number.isFinite(v)), true)
})

test('b8g-moe-gate: rejects non-finite or over-energy ingress tensors at boundary gate', () => {
  const router = new B8gVerifiedMoERouter({
    hiddenSize: 4,
    topK: 1,
    loopPasses: 1,
    maxIngressL2Norm: 10,
    maxEgressL2Norm: 10,
    loopBlendAlpha: 0.5,
  })
  const nanRes = router.dispatch(new Float32Array([1, Number.NaN, 0, 0]), {
    tenantId: 'frank',
    allowedScopes: ['*'],
  })
  assert.equal(nanRes.ok, false)
  assert.equal(nanRes.audits[0]?.status, 'NON_FINITE_INPUT')

  const normRes = router.dispatch(new Float32Array([100, 0, 0, 0]), {
    tenantId: 'frank',
    allowedScopes: ['*'],
  })
  assert.equal(normRes.ok, false)
  assert.equal(normRes.audits[0]?.status, 'INGRESS_NORM_EXCEEDED')
})

test('b8g-moe-gate: symbolic safe gate rejects violating expert and quarantines it', () => {
  const router = new B8gVerifiedMoERouter({
    hiddenSize: 4,
    topK: 2,
    loopPasses: 1,
    maxIngressL2Norm: 50,
    maxEgressL2Norm: 50,
    loopBlendAlpha: 0.5,
  })
  router.registerExpert({
    id: 7,
    name: 'bad-syntax-expert',
    requiredScope: 'public',
    routerKey: new Float32Array([2, 0, 0, 0]),
    forward: (inp, out) => out.set(inp),
    symbolicGate: () => ({ ok: false, reason: 'V8 syntax check failed' }),
  })
  router.registerExpert({
    id: 8,
    name: 'verified-expert',
    requiredScope: 'public',
    routerKey: new Float32Array([1, 0, 0, 0]),
    forward: (inp, out) => out.set(inp),
    symbolicGate: () => ({ ok: true }),
  })

  const res = router.dispatch(new Float32Array([1, 0, 0, 0]), {
    tenantId: 'frank',
    allowedScopes: ['public'],
  })
  assert.equal(res.ok, true)
  assert.equal(router.isQuarantined(7), true)
  assert.equal(res.activeExpertsExecuted, 1)
  assert.ok(res.audits.some((a) => a.status === 'SYMBOLIC_GATE_REJECTED'))
})
