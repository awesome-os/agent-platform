/**
 * @module
 * Block-Sparse Linear Operator with Zero-Copy Masking (`block-sparse.mts`).
 *
 * Implements structured block-sparse routing over 16x16 tile grids:
 *   1. Evaluates Frobenius energy norms for each [blockSize x blockSize] sub-block.
 *   2. Magnitude-prunes lowest energy tiles to hit strict layer sparsity targets.
 *   3. Tracks real-time active tile counts, actual sparsity, and theoretical FLOP reductions.
 *   4. Dispatches sparse matrix multiplication through the B8G microABI bridge without copying.
 */

import { B8GMicroABIBridge } from './microabi-bridge.mts'
import type { BlockSparseMetrics } from './types.mts'
import type { B8GModelArena } from './arena.mts'

export class BlockSparseLinear {
  public readonly inFeatures: number
  public readonly outFeatures: number
  public readonly blockSize: number
  public readonly sparsityTarget: number
  public readonly numBlocksH: number
  public readonly numBlocksW: number

  public readonly weight: Float32Array
  public readonly blockMask: Uint8Array // 1 = active, 0 = pruned

  constructor(
    arena: B8GModelArena,
    tensorPrefix: string,
    inFeatures: number,
    outFeatures: number,
    blockSize: number = 16,
    sparsityTarget: number = 0.5
  ) {
    if (inFeatures % blockSize !== 0 || outFeatures % blockSize !== 0) {
      throw new Error(`Features must be divisible by blockSize (${blockSize}): inFeatures=${inFeatures}, outFeatures=${outFeatures}`)
    }

    this.inFeatures = inFeatures
    this.outFeatures = outFeatures
    this.blockSize = blockSize
    this.sparsityTarget = sparsityTarget
    this.numBlocksH = outFeatures / blockSize
    this.numBlocksW = inFeatures / blockSize

    // Allocate / retrieve weight from B8G arena
    this.weight = arena.allocateTensor(`${tensorPrefix}.weight`, [outFeatures, inFeatures])

    // Allocate mask in linear memory
    this.blockMask = new Uint8Array(this.numBlocksH * this.numBlocksW)
    this.initializeWeights(0.02)
    this.updateSparsityMask()
  }

  /**
   * Initializes weights with deterministic pseudo-random gaussian values.
   */
  private initializeWeights(std: number): void {
    let seed = 1337
    const random = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296
      return seed / 4294967296
    }
    for (let i = 0; i < this.weight.length; i++) {
      // Box-Muller transform for normal distribution
      const u1 = Math.max(1e-7, random())
      const u2 = random()
      const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
      this.weight[i] = z * std
    }
  }

  /**
   * Computes Frobenius norm per tile and masks out lowest energy blocks.
   */
  public updateSparsityMask(): void {
    const totalBlocks = this.numBlocksH * this.numBlocksW
    const blockNorms: { index: number; norm: number }[] = []

    for (let bh = 0; bh < this.numBlocksH; bh++) {
      const rowBase = bh * this.blockSize
      for (let bw = 0; bw < this.numBlocksW; bw++) {
        const colBase = bw * this.blockSize
        let sumSq = 0.0

        for (let r = 0; r < this.blockSize; r++) {
          const rowOffset = (rowBase + r) * this.inFeatures + colBase
          for (let c = 0; c < this.blockSize; c++) {
            const v = this.weight[rowOffset + c]
            sumSq += v * v
          }
        }

        const blockIdx = bh * this.numBlocksW + bw
        blockNorms.push({ index: blockIdx, norm: Math.sqrt(sumSq) })
      }
    }

    // Sort ascending by energy
    blockNorms.sort((a, b) => a.norm - b.norm)

    const numPruned = Math.floor(totalBlocks * this.sparsityTarget)
    this.blockMask.fill(1)
    for (let i = 0; i < numPruned; i++) {
      this.blockMask[blockNorms[i].index] = 0
    }
  }

  /**
   * Computes block-sparse routing metrics.
   */
  public getMetrics(): BlockSparseMetrics {
    const totalBlocks = this.numBlocksH * this.numBlocksW
    let activeBlocks = 0
    for (let i = 0; i < totalBlocks; i++) {
      if (this.blockMask[i] === 1) activeBlocks++
    }
    const actualSparsity = 1.0 - (activeBlocks / totalBlocks)
    return {
      totalBlocks,
      activeBlocks,
      actualSparsity,
      theoreticalFlopReduction: actualSparsity * 100.0
    }
  }

  /**
   * Zero-copy forward pass using B8G microABI blockSparseGemm.
   */
  public forward(inVec: Float32Array, outVec: Float32Array): void {
    B8GMicroABIBridge.blockSparseGemm(
      inVec,
      this.weight,
      outVec,
      this.inFeatures,
      this.outFeatures,
      this.blockSize,
      this.blockMask
    )
  }
}
