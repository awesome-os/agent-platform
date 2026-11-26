/**
 * @file tutorials/01-stamp-boundary-verified-moe.mts
 * @description Tutorial 01: Deterministically stamp a Semantics-of-Operations model blueprint
 * (`b8g-sovereign-moe-v1`) and dispatch tensors through `B8gVerifiedMoERouter` with
 * Ingress/Egress boundary gates, CapabilityToken checks, expert quarantine, and Web Streams.
 */

import assert from 'node:assert/strict';
import {
  CURATED_MODEL_SPECS,
  getCuratedModelSpec,
  stampModelBlueprint,
} from '../packages/model-ops/index.mts';
import {
  B8gVerifiedMoERouter,
  createMoeGateTransform,
  type CapabilityToken,
  type VerifiedMoeExpert,
} from '../packages/moe-gate/index.mts';
import type { StreamFrame } from '../packages/schema/index.mts';

async function main(): Promise<void> {
  console.log('=== Tutorial 01: Stamp Model Blueprint & Boundary-Verified MoE ===');
  console.log(`Curated model specs available: ${CURATED_MODEL_SPECS.map((s) => s.id).join(', ')}`);

  const spec = getCuratedModelSpec('b8g-sovereign-moe-v1');
  assert.ok(spec, 'Expected curated spec b8g-sovereign-moe-v1');

  const blueprint = stampModelBlueprint(spec);
  console.log(`\nStamped Blueprint: ${blueprint.name}`);
  console.log(`  Physical -> Logical layers : ${blueprint.physicalLayers} -> ${blueprint.logicalLayers} (${blueprint.loopFactor}x loop)`);
  console.log(`  Estimated Total Params     : ${(blueprint.estimatedTotalParams / 1e9).toFixed(2)}B`);
  console.log(`  Estimated Active Params    : ${(blueprint.estimatedEffectiveActiveParams / 1e9).toFixed(2)}B`);
  console.log(`  Bits/Weight Avg            : ${blueprint.bitsPerWeightAvg} BPW`);
  console.log(`  Total 64B Arena Bytes      : ${blueprint.totalArenaBytes} B (${blueprint.arenaSlabs.length} slabs)`);
  for (const slab of blueprint.arenaSlabs) {
    assert.equal(slab.aligned64, true, `Slab ${slab.name} must be 64B aligned`);
    console.log(`    - ${slab.name.padEnd(22)} offset=${slab.offsetBytes} size=${slab.sizeBytes} aligned64=${slab.aligned64}`);
  }

  // Build Boundary-Verified MoE Router (topK=2, 2 loop passes, strict L2 bounds)
  const router = new B8gVerifiedMoERouter({
    hiddenSize: 4,
    topK: 2,
    loopPasses: 2,
    maxIngressL2Norm: 10.0,
    maxEgressL2Norm: 10.0,
    loopBlendAlpha: 0.5,
  });

  const healthyMathExpert: VerifiedMoeExpert = {
    id: 1,
    name: 'expert-math-stable',
    requiredScope: 'moe.route',
    routerKey: new Float32Array([1, 0, 0, 0]),
    forward(input, out) {
      for (let i = 0; i < out.length; i += 1) out[i] = input[i]! * 0.9;
    },
    symbolicGate(output) {
      return Number.isFinite(output[0]!) ? { ok: true } : { ok: false, reason: 'Non-finite slot 0' };
    },
  };

  const poisonedExpert: VerifiedMoeExpert = {
    id: 2,
    name: 'expert-poisoned-nan',
    requiredScope: 'moe.route',
    routerKey: new Float32Array([0.95, 0.2, 0, 0]),
    forward(_input, out) {
      out[0] = Number.NaN;
    },
  };

  const backupExpert: VerifiedMoeExpert = {
    id: 3,
    name: 'expert-backup-safe',
    requiredScope: 'moe.route',
    routerKey: new Float32Array([0.5, 0.5, 0, 0]),
    forward(input, out) {
      for (let i = 0; i < out.length; i += 1) out[i] = input[i]! * 0.5 + 0.1;
    },
  };

  router.registerExpert(healthyMathExpert);
  router.registerExpert(poisonedExpert);
  router.registerExpert(backupExpert);

  const token: CapabilityToken = {
    tenantId: 'tutorial-tenant',
    allowedScopes: ['moe.route'],
  };

  const x = new Float32Array([1.0, 0.25, -0.25, 0.5]);
  const res = router.dispatch(x, token);

  console.log('\nMoE Dispatch Result:');
  console.log(`  ok=${res.ok}, passes=${res.passesExecuted}, activeExperts=${res.activeExpertsExecuted}, quarantined=${res.quarantinedCount}`);
  console.log(`  poisonedExpert quarantined? ${router.isQuarantined(2)}`);
  assert.equal(res.ok, true);
  assert.equal(router.isQuarantined(2), true);
  assert.equal(Number.isFinite(res.output[0]!), true);

  // Stream a tensor frame through createMoeGateTransform
  const transform = createMoeGateTransform({ router, token });
  const writer = transform.writable.getWriter();
  const reader = transform.readable.getReader();

  const frame: StreamFrame<Float32Array> = {
    kind: 'data',
    source: 'tutorial-01',
    minPersona: 'developer',
    payload: new Float32Array([0.8, 0.4, 0.1, 0.0]),
    ts: Date.now(),
  };

  await writer.write(frame);
  await writer.close();
  const outFrame = await reader.read();
  assert.equal(outFrame.done, false);
  assert.equal(outFrame.value.kind, 'data');
  assert.equal(outFrame.value.payload.quarantinedCount, 0, 'Poisoned expert remains quarantined on subsequent dispatch');
  console.log('  StreamFrame dispatch verified: poisoned expert stayed quarantined across turns.');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
