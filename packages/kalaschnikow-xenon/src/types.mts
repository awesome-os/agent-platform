/**
 * @module
 * Types and interfaces for Model Kalaschnikow (v0.1-pre.1)
 * Asymmetric layer-wise tapered transformer architecture atop B8G microABI linear memory arena.
 */

export interface LayerGeometry {
  readonly layerIndex: number
  readonly dModelIn: number
  readonly dModelOut: number
  readonly nHead: number
  readonly dFf: number
  readonly sparsity: number
  readonly blockSize: number
}

export interface ModelArchitectureConfig {
  readonly name: string
  readonly architecture: 'symmetric' | 'asymmetric_tapered'
  readonly vocabSize: number
  readonly maxSeqLen: number
  readonly layers: readonly LayerGeometry[]
}

export interface ParameterBreakdown {
  readonly embeddingParams: number
  readonly attentionParams: number
  readonly mlpParams: number
  readonly projectionParams: number
  readonly normParams: number
  readonly headParams: number
  readonly totalParams: number
  readonly theoreticalFlopsPerToken: number
}

export interface BlockSparseMetrics {
  readonly totalBlocks: number
  readonly activeBlocks: number
  readonly actualSparsity: number
  readonly theoreticalFlopReduction: number
}

export interface LayerSparsityReport {
  readonly layerIndex: number
  readonly mlp1: BlockSparseMetrics
  readonly mlp2: BlockSparseMetrics
}

export interface ScalingDataPoint {
  readonly params: number // N
  readonly tokens: number // D
  readonly loss: number   // L
}

export interface ChinchillaParameters {
  readonly E: number
  readonly A: number
  readonly B: number
  readonly alpha: number
  readonly beta: number
}

export interface ExtrapolationTarget {
  readonly targetParams: number
  readonly optimalTokens: number
  readonly tokenToParamRatio: number
  readonly predictedLoss: number
  readonly estimatedTrainingFlops: number
}

export interface ArenaTensorDescriptor {
  readonly name: string
  readonly shape: readonly number[]
  readonly dtype: 'float32' | 'int32' | 'uint8'
  readonly offsetBytes: number
  readonly byteLength: number
  readonly elementCount: number
}

export interface ArenaHeader {
  readonly magic: string // 'B8G_ARENA_V1' (12 bytes)
  readonly version: number
  readonly tensorCount: number
  readonly totalBytes: number
}
