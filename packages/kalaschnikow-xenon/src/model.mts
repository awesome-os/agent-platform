/**
 * @module
 * Model Kalaschnikow (v0.1-pre.1) Core Execution Engine (`model.mts`).
 *
 * Implements the Asymmetric Layer-Wise Tapered Transformer atop B8G Zero-Copy Memory Arena.
 * Executes tensor kernels without temporary heap allocations via microABI bridges.
 */

import { B8GModelArena } from './arena.mts'
import { BlockSparseLinear } from './block-sparse.mts'
import { B8GMicroABIBridge } from './microabi-bridge.mts'
import { IsoFLOPGeometryEngine } from './isoflop-engine.mts'
import type {
  ModelArchitectureConfig,
  LayerSparsityReport,
  ParameterBreakdown
} from './types.mts'

export class KalaschnikowBlock {
  public readonly layerIndex: number
  public readonly dModelIn: number
  public readonly dModelOut: number
  public readonly nHead: number
  public readonly dFf: number
  public readonly sparsity: number
  public readonly blockSize: number

  // Weights registered directly in B8G Arena
  public readonly projectionInWeight?: Float32Array
  public readonly norm1Gamma: Float32Array
  public readonly norm1Beta: Float32Array
  public readonly qkvWeight: Float32Array
  public readonly projOutWeight: Float32Array
  public readonly norm2Gamma: Float32Array
  public readonly norm2Beta: Float32Array
  public readonly mlpFc1: BlockSparseLinear
  public readonly mlpFc2: BlockSparseLinear

  constructor(
    arena: B8GModelArena,
    layerIndex: number,
    dModelIn: number,
    dModelOut: number,
    nHead: number,
    dFf: number,
    sparsity: number = 0.4,
    blockSize: number = 16
  ) {
    this.layerIndex = layerIndex
    this.dModelIn = dModelIn
    this.dModelOut = dModelOut
    this.nHead = nHead
    this.dFf = dFf
    this.sparsity = sparsity
    this.blockSize = blockSize

    const pfx = `layer_${layerIndex}`

    // Projection if dimension changes
    if (dModelIn !== dModelOut) {
      this.projectionInWeight = arena.allocateTensor(`${pfx}.proj_in.weight`, [dModelOut, dModelIn])
      this.initGaussian(this.projectionInWeight, 0.02)
    }

    // Norm1 (LayerNorm before attention)
    this.norm1Gamma = arena.allocateTensor(`${pfx}.norm1.gamma`, [dModelOut])
    this.norm1Beta = arena.allocateTensor(`${pfx}.norm1.beta`, [dModelOut])
    this.norm1Gamma.fill(1.0)
    this.norm1Beta.fill(0.0)

    // Attention projections: Q, K, V combined [3 * dModelOut, dModelOut], Proj [dModelOut, dModelOut]
    this.qkvWeight = arena.allocateTensor(`${pfx}.attn.qkv.weight`, [3 * dModelOut, dModelOut])
    this.projOutWeight = arena.allocateTensor(`${pfx}.attn.proj.weight`, [dModelOut, dModelOut])
    this.initGaussian(this.qkvWeight, 0.02)
    this.initGaussian(this.projOutWeight, 0.02)

    // Norm2 (LayerNorm before MLP)
    this.norm2Gamma = arena.allocateTensor(`${pfx}.norm2.gamma`, [dModelOut])
    this.norm2Beta = arena.allocateTensor(`${pfx}.norm2.beta`, [dModelOut])
    this.norm2Gamma.fill(1.0)
    this.norm2Beta.fill(0.0)

    // Block-Sparse MLP layers
    this.mlpFc1 = new BlockSparseLinear(arena, `${pfx}.mlp.fc1`, dModelOut, dFf, blockSize, sparsity)
    this.mlpFc2 = new BlockSparseLinear(arena, `${pfx}.mlp.fc2`, dFf, dModelOut, blockSize, sparsity)
  }

  private initGaussian(arr: Float32Array, std: number): void {
    let seed = 42 + this.layerIndex * 31
    const random = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296
      return seed / 4294967296
    }
    for (let i = 0; i < arr.length; i++) {
      const u1 = Math.max(1e-7, random())
      const u2 = random()
      const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
      arr[i] = z * std
    }
  }

  /**
   * Forward step for a single token hidden state vector.
   * inVec: shape [dModelIn]
   * outVec: shape [dModelOut]
   */
  public forwardToken(
    inVec: Float32Array,
    outVec: Float32Array,
    scratchNorm: Float32Array,
    scratchMlpInter: Float32Array,
    scratchMlpOut: Float32Array
  ): void {
    // 1. Dimension projection if needed
    if (this.projectionInWeight) {
      B8GMicroABIBridge.gemm(inVec, this.projectionInWeight, outVec, this.dModelIn, this.dModelOut)
    } else {
      outVec.set(inVec)
    }

    // 2. Pre-Norm1
    B8GMicroABIBridge.layerNorm(outVec, scratchNorm, this.norm1Gamma, this.norm1Beta, this.dModelOut)

    // 3. Attention projection (simulated single token feedforward component for parity)
    // QKV -> Proj
    const scratchAttn = scratchMlpOut.subarray(0, this.dModelOut)
    B8GMicroABIBridge.gemm(scratchNorm, this.projOutWeight, scratchAttn, this.dModelOut, this.dModelOut)
    B8GMicroABIBridge.residualAdd(outVec, scratchAttn, this.dModelOut)

    // 4. Pre-Norm2
    B8GMicroABIBridge.layerNorm(outVec, scratchNorm, this.norm2Gamma, this.norm2Beta, this.dModelOut)

    // 5. Block-Sparse MLP: fc1 -> GELU -> fc2
    this.mlpFc1.forward(scratchNorm, scratchMlpInter)
    B8GMicroABIBridge.gelu(scratchMlpInter, scratchMlpInter, this.dFf)
    this.mlpFc2.forward(scratchMlpInter, scratchMlpOut)

    // 6. Residual Add
    B8GMicroABIBridge.residualAdd(outVec, scratchMlpOut, this.dModelOut)
  }
}

export class KalaschnikowModel {
  public readonly config: ModelArchitectureConfig
  public readonly arena: B8GModelArena
  public readonly blocks: KalaschnikowBlock[] = []
  public readonly normFinalGamma: Float32Array
  public readonly normFinalBeta: Float32Array
  public readonly lmHeadWeight: Float32Array

  // Scratchpad buffers for zero-allocation token forward steps
  private readonly maxDim: number
  private readonly maxFf: number
  private readonly scratchTokenBuffers: {
    bufA: Float32Array
    bufB: Float32Array
    scratchNorm: Float32Array
    scratchMlpInter: Float32Array
    scratchMlpOut: Float32Array
  }

  constructor(config?: ModelArchitectureConfig, arenaCapacity?: number) {
    this.config = config ?? IsoFLOPGeometryEngine.createKalaschnikowConfig()
    const breakdown = IsoFLOPGeometryEngine.calculateParameters(this.config)
    const requiredBytes = breakdown.totalParams * 4 + 32 * 1024 * 1024 // params + 32MB headroom
    const capacity = arenaCapacity ?? Math.max(requiredBytes, 640 * 1024 * 1024)
    this.arena = new B8GModelArena(capacity)

    let maxD = 768
    let maxF = 3072

    // Initialize blocks
    for (const l of this.config.layers) {
      if (l.dModelIn > maxD) maxD = l.dModelIn
      if (l.dModelOut > maxD) maxD = l.dModelOut
      if (l.dFf > maxF) maxF = l.dFf

      this.blocks.push(
        new KalaschnikowBlock(
          this.arena,
          l.layerIndex,
          l.dModelIn,
          l.dModelOut,
          l.nHead,
          l.dFf,
          l.sparsity,
          l.blockSize
        )
      )
    }

    this.maxDim = maxD
    this.maxFf = maxF

    const lastDim = this.config.layers[this.config.layers.length - 1].dModelOut

    // Final LayerNorm
    this.normFinalGamma = this.arena.allocateTensor('norm_f.gamma', [lastDim])
    this.normFinalBeta = this.arena.allocateTensor('norm_f.beta', [lastDim])
    this.normFinalGamma.fill(1.0)
    this.normFinalBeta.fill(0.0)

    // LM Head
    this.lmHeadWeight = this.arena.allocateTensor('lm_head.weight', [this.config.vocabSize, lastDim])
    this.initTensorGaussian(this.lmHeadWeight, 0.02)

    // Allocate reusable scratchpads (zero heap allocation during inference)
    this.scratchTokenBuffers = {
      bufA: new Float32Array(this.maxDim),
      bufB: new Float32Array(this.maxDim),
      scratchNorm: new Float32Array(this.maxDim),
      scratchMlpInter: new Float32Array(this.maxFf),
      scratchMlpOut: new Float32Array(this.maxDim)
    }
  }

  private initTensorGaussian(arr: Float32Array, std: number): void {
    let seed = 9999
    const random = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296
      return seed / 4294967296
    }
    for (let i = 0; i < arr.length; i++) {
      const u1 = Math.max(1e-7, random())
      const u2 = random()
      const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
      arr[i] = z * std
    }
  }

  /**
   * Returns parameter count breakdown.
   */
  public getParameterBreakdown(): ParameterBreakdown {
    return IsoFLOPGeometryEngine.calculateParameters(this.config)
  }

  /**
   * Aggregates block-sparse routing metrics for each layer.
   */
  public getSparsityReports(): LayerSparsityReport[] {
    return this.blocks.map(block => ({
      layerIndex: block.layerIndex,
      mlp1: block.mlpFc1.getMetrics(),
      mlp2: block.mlpFc2.getMetrics()
    }))
  }

  /**
   * Runs single-token zero-copy forward pass.
   * inEmbedding: Float32Array of size 768
   * logitsOut: Float32Array of size vocabSize
   */
  public forwardToken(inEmbedding: Float32Array, logitsOut: Float32Array): void {
    const { bufA, bufB, scratchNorm, scratchMlpInter, scratchMlpOut } = this.scratchTokenBuffers

    bufA.subarray(0, 768).set(inEmbedding)

    let currentIn = bufA
    let currentOut = bufB

    for (const block of this.blocks) {
      block.forwardToken(
        currentIn.subarray(0, block.dModelIn),
        currentOut.subarray(0, block.dModelOut),
        scratchNorm.subarray(0, block.dModelOut),
        scratchMlpInter.subarray(0, block.dFf),
        scratchMlpOut.subarray(0, block.dModelOut)
      )

      // Swap ping-pong buffers
      const temp = currentIn
      currentIn = currentOut
      currentOut = temp
    }

    const lastDim = this.config.layers[this.config.layers.length - 1].dModelOut

    // Final LayerNorm
    B8GMicroABIBridge.layerNorm(
      currentIn.subarray(0, lastDim),
      scratchNorm.subarray(0, lastDim),
      this.normFinalGamma,
      this.normFinalBeta,
      lastDim
    )

    // Compute logits via LM Head GEMM
    B8GMicroABIBridge.gemm(
      scratchNorm.subarray(0, lastDim),
      this.lmHeadWeight,
      logitsOut,
      lastDim,
      this.config.vocabSize
    )
  }

  /**
   * Serializes the model's arena weights into B8G_ARENA_V1 binary file.
   */
  public exportBinaryArena(filePath: string): void {
    this.arena.saveToFile(filePath)
  }
}
