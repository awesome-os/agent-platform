#!/usr/bin/env node
/**
 * @file intersections/b8g-moe-gate/cli.mts
 * @description Unified CLI (`.\bin\b8g-moe`) for the Awesome-OS Model Atlas,
 * Blueprint Stamper (`sections/b8g-model-ops`), and Boundary-Verified MoE Gate Engine.
 */

import {
  CURATED_MODEL_SPECS,
  getCuratedModelSpec,
  stampModelBlueprint,
} from '../../sections/b8g-model-ops/index.mts'
import { B8gVerifiedMoERouter } from './index.mts'

const args = process.argv.slice(2)
const cmd = args[0] ?? 'catalog'

function fmtGb(bytes: number): string {
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
}

function fmtB(params: number): string {
  return `${(params / 1e9).toFixed(2)}B`
}

if (cmd === 'catalog') {
  console.log('\n  B8G Unified Model Architecture Atlas (sections/b8g-model-ops)\n')
  for (const spec of CURATED_MODEL_SPECS) {
    const bp = stampModelBlueprint(spec)
    const refNote = bp.verifiedReference
      ? ` [ref: ${bp.verifiedReference.totalParamsB}B tot / ${bp.verifiedReference.effectiveActiveParamsB}B eff / ${bp.verifiedReference.footprintGb} GB]`
      : ''
    console.log(
      `  • ${spec.id.padEnd(22)} | ${bp.physicalLayers}→${bp.logicalLayers}L | ${spec.quant.padEnd(17)} | gate=${bp.boundaryVerified ? 'VERIFIED' : 'none    '} | ${spec.name}${refNote}`
    )
  }
  console.log('\n  Usage: .\\bin\\b8g-moe.ps1 catalog | stamp <preset-id> | verify\n')
} else if (cmd === 'stamp') {
  const id = args[1] ?? 'b8g-sovereign-moe-v1'
  const spec = getCuratedModelSpec(id)
  if (!spec) {
    console.error(`Unknown preset '${id}'. Available: ${CURATED_MODEL_SPECS.map((s) => s.id).join(', ')}`)
    process.exit(1)
  }
  const bp = stampModelBlueprint(spec)
  console.log(JSON.stringify({
    specId: bp.specId,
    name: bp.name,
    physicalLayers: bp.physicalLayers,
    logicalLayers: bp.logicalLayers,
    loopFactor: bp.loopFactor,
    analyticalParams: {
      total: fmtB(bp.estimatedTotalParams),
      baseActive: fmtB(bp.estimatedBaseActiveParams),
      effectiveActive: fmtB(bp.estimatedEffectiveActiveParams),
    },
    verifiedReference: bp.verifiedReference ?? null,
    quantization: { profile: spec.quant, bitsPerWeightAvg: bp.bitsPerWeightAvg, weightBytes: fmtGb(bp.estimatedWeightBytes) },
    boundaryVerified: bp.boundaryVerified,
    arenaSlabs: bp.arenaSlabs,
    totalArenaBytes: bp.totalArenaBytes,
  }, null, 2))
} else if (cmd === 'verify') {
  const router = new B8gVerifiedMoERouter({
    hiddenSize: 8,
    topK: 2,
    loopPasses: 3,
    maxIngressL2Norm: 100,
    maxEgressL2Norm: 100,
    loopBlendAlpha: 0.5,
  })

  router.registerExpert({
    id: 0,
    name: 'coding-expert-ok',
    requiredScope: 'public:code',
    routerKey: new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]),
    forward: (inp, out) => {
      for (let i = 0; i < inp.length; i += 1) out[i] = inp[i]! * 0.9
    },
  })
  router.registerExpert({
    id: 1,
    name: 'rogue-nan-expert',
    requiredScope: 'public:code',
    routerKey: new Float32Array([0.95, 0, 0, 0, 0, 0, 0, 0]),
    forward: (_inp, out) => {
      out[0] = Number.NaN
    },
  })
  router.registerExpert({
    id: 2,
    name: 'private-vault-expert',
    requiredScope: 'tenant:vault',
    routerKey: new Float32Array([10, 0, 0, 0, 0, 0, 0, 0]),
    forward: (inp, out) => {
      out.set(inp)
    },
  })

  const x = new Float32Array([1, 0.5, 0.25, 0, 0, 0, 0, 0])
  const res = router.dispatch(x, { tenantId: 'guest', allowedScopes: ['public:code'] })
  const allFinite = res.output.every((v) => Number.isFinite(v))
  console.log(
    `[b8g-moe verify] ok=${res.ok} passes=${res.passesExecuted} active=${res.activeExpertsExecuted} ` +
    `quarantined=${res.quarantinedCount} capDenials=${res.capabilityDenials} outputFinite=${allFinite}`
  )
  if (!res.ok || !allFinite || res.quarantinedCount < 1 || res.capabilityDenials < 1) {
    process.exit(1)
  }
} else {
  console.error(`Unknown command '${cmd}'. Use: catalog | stamp <preset-id> | verify`)
  process.exit(1)
}
