/**
 * @file sections/b8g-model-ops/index.mts
 * @description Canonical Semantics of Operations IR, Curated Model Preset Registry,
 * and Deterministic Model Blueprint Stamper for Awesome-OS (`B8G-SECT-1`).
 *
 * Strictly `erasableSyntaxOnly: true` & zero-dependency compliant.
 */

export type TopologyOp =
  | { readonly kind: 'standard_stack'; readonly layers: number }
  | {
      readonly kind: 'looped_core'
      readonly preludeLayers: number
      readonly coreLayers: number
      readonly loopPasses: number
      readonly codaLayers: number
      readonly gateRank: number
    }
  | {
      readonly kind: 'asymmetric_tapered'
      readonly layerDims: readonly number[]
    }

export type ResidualOp =
  | { readonly kind: 'additive' }
  | { readonly kind: 'block_attention_residual'; readonly blockSize: number }

export type AttentionOp =
  | { readonly kind: 'gqa'; readonly qHeads: number; readonly kvHeads: number }
  | {
      readonly kind: 'hybrid_local_global'
      readonly qHeads: number
      readonly kvHeads: number
      readonly localWindow: number
      readonly localToGlobalRatio: number
    }
  | { readonly kind: 'mla'; readonly qHeads: number; readonly kvLoraRank: number }

export type FfnOp =
  | { readonly kind: 'dense'; readonly intermediateSize: number }
  | {
      readonly kind: 'moe_routed'
      readonly activeExperts: number
      readonly totalExperts: number
      readonly sharedExperts: number
      readonly expertIntermediateSize: number
    }
  | {
      readonly kind: 'block_sparse'
      readonly intermediateSize: number
      readonly sparsity: number
      readonly blockSize: number
    }

export type QuantProfile =
  | 'fp16'
  | 'mxfp8'
  | 'nvfp4'
  | 'q4_k_m'
  | 'bitnet_158'
  | 'littlebit_sub1'
  | 'mixed_nvfp4_mxfp8'

export type MemoryOp =
  | { readonly kind: 'kv_cache_only' }
  | { readonly kind: 'per_loop_kv_cache' }
  | {
      readonly kind: 'atlaskv_hikvp'
      readonly clusters: number
      readonly topKClusters: number
      readonly capabilityIsolated: boolean
    }

export type SpeculatorOp =
  | { readonly kind: 'none' }
  | {
      readonly kind: 'dspark'
      readonly markovRank: number
      readonly hasConfidenceHead: boolean
      readonly overheadBytes: number
    }

export type BoundaryGateOp =
  | { readonly kind: 'none' }
  | {
      readonly kind: 'verified_moe_gate'
      readonly requireCapabilityToken: boolean
      readonly maxIngressL2Norm: number
      readonly maxEgressL2Norm: number
      readonly quarantineOnViolation: boolean
      readonly enableSymbolicSafeGate: boolean
    }

export interface ModelOpSpec {
  readonly id: string
  readonly name: string
  readonly origin: string
  readonly license: string
  readonly docPath: string
  readonly hiddenSize: number
  readonly vocabSize: number
  readonly maxContextTokens: number
  readonly multimodal: boolean
  readonly topology: TopologyOp
  readonly residual: ResidualOp
  readonly attention: AttentionOp
  readonly ffn: FfnOp
  readonly quant: QuantProfile
  readonly memory: MemoryOp
  readonly speculator: SpeculatorOp
  readonly boundaryGate: BoundaryGateOp
  /** Optional verified external parameter overrides (when exact config differs from analytical estimate). */
  readonly verifiedReference?: {
    readonly totalParamsB: number
    readonly baseActiveParamsB: number
    readonly effectiveActiveParamsB: number
    readonly footprintGb: number
  }
}

export interface ArenaSlabDescriptor {
  readonly name: string
  readonly offsetBytes: number
  readonly sizeBytes: number
  readonly aligned64: boolean
}

export interface StampedModelBlueprint {
  readonly specId: string
  readonly name: string
  readonly physicalLayers: number
  readonly logicalLayers: number
  readonly loopFactor: number
  readonly estimatedTotalParams: number
  readonly estimatedBaseActiveParams: number
  readonly estimatedEffectiveActiveParams: number
  readonly bitsPerWeightAvg: number
  readonly estimatedWeightBytes: number
  readonly arenaSlabs: readonly ArenaSlabDescriptor[]
  readonly totalArenaBytes: number
  readonly boundaryVerified: boolean
  readonly verifiedReference?: ModelOpSpec['verifiedReference']
}

const BPW_TABLE: Readonly<Record<QuantProfile, number>> = {
  fp16: 16.0,
  mxfp8: 8.25,
  nvfp4: 4.25,
  q4_k_m: 4.5,
  bitnet_158: 1.58,
  littlebit_sub1: 0.25,
  mixed_nvfp4_mxfp8: 5.1,
}

function align64(n: number): number {
  return (n + 63) & ~63
}

/**
 * Deterministically stamps a ModelOpSpec into an executable Blueprint with exact
 * physical/logical depth, parameter budgets, quantization footprint, and 64B-aligned
 * `B8G_ARENA_V1` slab offsets.
 */
export function stampModelBlueprint(spec: ModelOpSpec): StampedModelBlueprint {
  let physicalLayers = 0
  let logicalLayers = 0
  let loopFactor = 1

  if (spec.topology.kind === 'standard_stack') {
    physicalLayers = spec.topology.layers
    logicalLayers = spec.topology.layers
  } else if (spec.topology.kind === 'looped_core') {
    physicalLayers = spec.topology.preludeLayers + spec.topology.coreLayers + spec.topology.codaLayers
    logicalLayers =
      spec.topology.preludeLayers +
      spec.topology.coreLayers * spec.topology.loopPasses +
      spec.topology.codaLayers
    loopFactor = logicalLayers / Math.max(1, physicalLayers)
  } else {
    physicalLayers = spec.topology.layerDims.length
    logicalLayers = spec.topology.layerDims.length
  }

  const h = spec.hiddenSize
  const embedParams = spec.vocabSize * h * 2 // input + lm_head

  // Attention params per layer
  let attnParamsPerLayer = 0
  if (spec.attention.kind === 'gqa' || spec.attention.kind === 'hybrid_local_global') {
    const headDim = Math.max(64, Math.floor(h / Math.max(1, spec.attention.qHeads)))
    const qProj = h * (spec.attention.qHeads * headDim)
    const kvProj = 2 * h * (spec.attention.kvHeads * headDim)
    const oProj = (spec.attention.qHeads * headDim) * h
    attnParamsPerLayer = qProj + kvProj + oProj
  } else {
    // MLA compressed KV
    attnParamsPerLayer = h * h * 2 + h * spec.attention.kvLoraRank * 2
  }

  // FFN params per layer (SwiGLU 3-matrix convention: gate, up, down)
  let ffnTotalPerLayer = 0
  let ffnActivePerLayer = 0
  if (spec.ffn.kind === 'dense') {
    ffnTotalPerLayer = 3 * h * spec.ffn.intermediateSize
    ffnActivePerLayer = ffnTotalPerLayer
  } else if (spec.ffn.kind === 'moe_routed') {
    const oneExpert = 3 * h * spec.ffn.expertIntermediateSize
    const router = h * spec.ffn.totalExperts
    ffnTotalPerLayer = router + (spec.ffn.totalExperts + spec.ffn.sharedExperts) * oneExpert
    ffnActivePerLayer = router + (spec.ffn.activeExperts + spec.ffn.sharedExperts) * oneExpert
  } else {
    const denseEq = 3 * h * spec.ffn.intermediateSize
    ffnTotalPerLayer = denseEq
    ffnActivePerLayer = Math.round(denseEq * (1 - spec.ffn.sparsity))
  }

  // Residual / loop controller overhead
  let extraPerLayer = 2 * h // RMSNorms
  if (spec.topology.kind === 'looped_core') {
    extraPerLayer += 2 * h * spec.topology.gateRank
  }
  if (spec.residual.kind === 'block_attention_residual') {
    extraPerLayer += h * spec.residual.blockSize
  }

  const estimatedTotalParams =
    embedParams + physicalLayers * (attnParamsPerLayer + ffnTotalPerLayer + extraPerLayer)
  const estimatedBaseActiveParams =
    embedParams + physicalLayers * (attnParamsPerLayer + ffnActivePerLayer + extraPerLayer)
  const estimatedEffectiveActiveParams =
    embedParams + logicalLayers * (attnParamsPerLayer + ffnActivePerLayer + extraPerLayer)

  const bitsPerWeightAvg = BPW_TABLE[spec.quant]
  const speculatorBytes = spec.speculator.kind === 'dspark' ? spec.speculator.overheadBytes : 0
  const estimatedWeightBytes =
    Math.ceil((estimatedTotalParams * bitsPerWeightAvg) / 8) + speculatorBytes

  // Build 64B-aligned B8G_ARENA_V1 slab layout
  const headerBytes = 64
  const routerAuditBytes = align64(logicalLayers * 32)
  const kvBudgetBytes = align64(logicalLayers * h * 4 * 4) // compact working-set slab per active slot
  const weightSlabBytes = align64(estimatedWeightBytes)
  const residualSlabBytes = align64(h * 4 * (spec.residual.kind === 'block_attention_residual' ? spec.residual.blockSize + 1 : 2))

  let cursor = 0
  const slabs: ArenaSlabDescriptor[] = []
  const pushSlab = (name: string, sizeBytes: number): void => {
    const aligned = align64(sizeBytes)
    slabs.push({ name, offsetBytes: cursor, sizeBytes: aligned, aligned64: cursor % 64 === 0 && aligned % 64 === 0 })
    cursor += aligned
  }

  pushSlab('b8g_model_header_v1', headerBytes)
  pushSlab('b8g_moe_gate_audit', routerAuditBytes)
  pushSlab('b8g_residual_stream', residualSlabBytes)
  pushSlab('b8g_kv_working_slab', kvBudgetBytes)
  pushSlab('b8g_weights_cas_view', weightSlabBytes)

  return {
    specId: spec.id,
    name: spec.name,
    physicalLayers,
    logicalLayers,
    loopFactor: Number(loopFactor.toFixed(2)),
    estimatedTotalParams,
    estimatedBaseActiveParams,
    estimatedEffectiveActiveParams,
    bitsPerWeightAvg,
    estimatedWeightBytes,
    arenaSlabs: slabs,
    totalArenaBytes: cursor,
    boundaryVerified: spec.boundaryGate.kind === 'verified_moe_gate',
    verifiedReference: spec.verifiedReference,
  }
}

/**
 * Curated Registry of Model Architecture Specs in Awesome-OS (`docs/models/`).
 */
export const CURATED_MODEL_SPECS: readonly ModelOpSpec[] = [
  {
    id: 'inkling-2-mini',
    name: 'Inkling 2 Mini (Thinking Machines Lab)',
    origin: 'thinkingmachines/Inkling-2-Mini',
    license: 'Apache-2.0',
    docPath: 'docs/models/01-inkling-2-mini-looped-moe.md',
    hiddenSize: 2304,
    vocabSize: 201024,
    maxContextTokens: 1_000_000,
    multimodal: true,
    topology: {
      kind: 'looped_core',
      preludeLayers: 6,
      coreLayers: 12,
      loopPasses: 3,
      codaLayers: 6,
      gateRank: 64,
    },
    residual: { kind: 'block_attention_residual', blockSize: 4 },
    attention: {
      kind: 'hybrid_local_global',
      qHeads: 24,
      kvHeads: 8,
      localWindow: 512,
      localToGlobalRatio: 5,
    },
    ffn: {
      kind: 'moe_routed',
      activeExperts: 8,
      totalExperts: 128,
      sharedExperts: 1,
      expertIntermediateSize: 1152,
    },
    quant: 'mixed_nvfp4_mxfp8',
    memory: { kind: 'per_loop_kv_cache' },
    speculator: {
      kind: 'dspark',
      markovRank: 64,
      hasConfidenceHead: true,
      overheadBytes: 1_200_000_000,
    },
    boundaryGate: { kind: 'none' },
    verifiedReference: {
      totalParamsB: 26.4,
      baseActiveParamsB: 3.4,
      effectiveActiveParamsB: 5.4,
      footprintGb: 20.0,
    },
  },
  {
    id: 'b8g-xenon-1b',
    name: 'B8G Xenon 1B (Kalaschnikow Tapered + AtlasKV)',
    origin: 'repository/model-kalaschnikow-v0.1-pre.1 + sections/atlas-kv',
    license: 'Awesome-OS Sovereign',
    docPath: 'docs/models/02-b8g-xenon-1b-kalaschnikow-atlaskv.md',
    hiddenSize: 1536,
    vocabSize: 65536,
    maxContextTokens: 131072,
    multimodal: false,
    topology: {
      kind: 'asymmetric_tapered',
      layerDims: [1536, 1536, 1536, 1536, 1280, 1280, 1280, 1280, 1024, 1024, 1024, 1024, 768, 768, 768, 768],
    },
    residual: { kind: 'additive' },
    attention: { kind: 'gqa', qHeads: 16, kvHeads: 4 },
    ffn: { kind: 'block_sparse', intermediateSize: 4096, sparsity: 0.5, blockSize: 16 },
    quant: 'q4_k_m',
    memory: { kind: 'atlaskv_hikvp', clusters: 128, topKClusters: 2, capabilityIsolated: true },
    speculator: { kind: 'none' },
    boundaryGate: {
      kind: 'verified_moe_gate',
      requireCapabilityToken: true,
      maxIngressL2Norm: 1e4,
      maxEgressL2Norm: 1e4,
      quarantineOnViolation: true,
      enableSymbolicSafeGate: false,
    },
  },
  {
    id: 'b8g-edge-littlebit',
    name: 'B8G Edge Router (LittleBit / NanoQuant 0.25 BPW)',
    origin: 'Samsung Research LittleBit / NanoQuant + Awesome-OS Edge',
    license: 'Research / Apache-2.0 Runtime',
    docPath: 'docs/models/03-sub-1-bit-littlebit-nanoquant.md',
    hiddenSize: 2048,
    vocabSize: 65536,
    maxContextTokens: 32768,
    multimodal: false,
    topology: { kind: 'standard_stack', layers: 24 },
    residual: { kind: 'additive' },
    attention: { kind: 'gqa', qHeads: 16, kvHeads: 4 },
    ffn: { kind: 'dense', intermediateSize: 5504 },
    quant: 'littlebit_sub1',
    memory: { kind: 'kv_cache_only' },
    speculator: { kind: 'none' },
    boundaryGate: {
      kind: 'verified_moe_gate',
      requireCapabilityToken: true,
      maxIngressL2Norm: 1e3,
      maxEgressL2Norm: 1e3,
      quarantineOnViolation: true,
      enableSymbolicSafeGate: false,
    },
  },
  {
    id: 'b8g-sovereign-moe-v1',
    name: 'B8G Sovereign Boundary-Verified MoE v1',
    origin: 'Awesome-OS Research Blueprint (Looped Core + Block AttnRes + AtlasKV + Safe Gates)',
    license: 'Awesome-OS Sovereign',
    docPath: 'docs/models/04-boundary-verified-moe-blueprint.md',
    hiddenSize: 2048,
    vocabSize: 131072,
    maxContextTokens: 1_000_000,
    multimodal: true,
    topology: {
      kind: 'looped_core',
      preludeLayers: 4,
      coreLayers: 8,
      loopPasses: 3,
      codaLayers: 4,
      gateRank: 64,
    },
    residual: { kind: 'block_attention_residual', blockSize: 4 },
    attention: {
      kind: 'hybrid_local_global',
      qHeads: 16,
      kvHeads: 4,
      localWindow: 1024,
      localToGlobalRatio: 3,
    },
    ffn: {
      kind: 'moe_routed',
      activeExperts: 4,
      totalExperts: 64,
      sharedExperts: 1,
      expertIntermediateSize: 1024,
    },
    quant: 'mixed_nvfp4_mxfp8',
    memory: { kind: 'atlaskv_hikvp', clusters: 128, topKClusters: 2, capabilityIsolated: true },
    speculator: {
      kind: 'dspark',
      markovRank: 32,
      hasConfidenceHead: true,
      overheadBytes: 512_000_000,
    },
    boundaryGate: {
      kind: 'verified_moe_gate',
      requireCapabilityToken: true,
      maxIngressL2Norm: 1e4,
      maxEgressL2Norm: 1e4,
      quarantineOnViolation: true,
      enableSymbolicSafeGate: true,
    },
  },
]

export function getCuratedModelSpec(id: string): ModelOpSpec | undefined {
  return CURATED_MODEL_SPECS.find((s) => s.id === id)
}
