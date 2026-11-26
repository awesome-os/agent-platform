/**
 * @module
 * IsoFLOP Parity & Geometry Engine for Model Kalaschnikow (`isoflop-engine.mts`).
 *
 * Implements strict compute budget balancing ($C \approx 6ND$):
 *   - Uniform 80M Symmetric Baseline (constant d_model=768, d_ff=3072, 12 layers).
 *   - Kalaschnikow Asymmetric Tapered Geometry (narrow ingestion -> wide reasoning core -> sparse top).
 *   - Parameter breakdown analysis & IsoFLOP parity verification (within +/-2% parameter envelope).
 */

import type { LayerGeometry, ModelArchitectureConfig, ParameterBreakdown } from './types.mts'

export class IsoFLOPGeometryEngine {
  /**
   * Builds the standard 80M uniform symmetric transformer config.
   */
  public static createSymmetricConfig(
    vocabSize: number = 32000,
    maxSeqLen: number = 2048
  ): ModelArchitectureConfig {
    const layers: LayerGeometry[] = []
    for (let i = 0; i < 12; i++) {
      layers.push({
        layerIndex: i,
        dModelIn: 768,
        dModelOut: 768,
        nHead: 12,
        dFf: 3072,
        sparsity: 0.0,
        blockSize: 16
      })
    }

    return {
      name: 'symmetric_baseline_80m',
      architecture: 'symmetric',
      vocabSize,
      maxSeqLen,
      layers
    }
  }

  /**
   * Builds the Kalaschnikow Asymmetric Tapered geometry config.
   * Dynamically balances intermediate dimensions (dFf) so that the total
   * parameter budget matches the symmetric baseline under strict IsoFLOP parity (+/- 0.5%).
   */
  public static createKalaschnikowConfig(
    vocabSize: number = 32000,
    maxSeqLen: number = 2048,
    autoBalanceIsoFlop: boolean = true
  ): ModelArchitectureConfig {
    const dims =  [512,  512,  640,  640,  768,  768,  896,  896, 1024, 1024,  768,  768]
    const heads = [  8,    8,   10,   10,   12,   12,   14,   14,   16,   16,   12,   12]
    // Raw initial d_ff ratios: 4x width
    let dFfs =   [2048, 2048, 2560, 2560, 3072, 3072, 3584, 3584, 4096, 4096, 3072, 3072]

    const buildConfig = (currFfs: number[]): ModelArchitectureConfig => {
      const layers: LayerGeometry[] = []
      let currDim = 768
      for (let i = 0; i < 12; i++) {
        const nextDim = dims[i]
        const sparsity = i < 4 ? 0.5 : (i < 8 ? 0.3 : 0.2)
        layers.push({
          layerIndex: i,
          dModelIn: currDim,
          dModelOut: nextDim,
          nHead: heads[i],
          dFf: currFfs[i],
          sparsity,
          blockSize: 16
        })
        currDim = nextDim
      }
      return {
        name: 'model_kalaschnikow_v0.1_pre.1',
        architecture: 'asymmetric_tapered',
        vocabSize,
        maxSeqLen,
        layers
      }
    }

    if (autoBalanceIsoFlop) {
      const baselineConfig = this.createSymmetricConfig(vocabSize, maxSeqLen)
      const targetParams = this.calculateParameters(baselineConfig).totalParams

      // Iteratively scale dFf to reach exact IsoFLOP target parity within +/-0.5%
      for (let iter = 0; iter < 10; iter++) {
        const candidate = buildConfig(dFfs)
        const currentParams = this.calculateParameters(candidate).totalParams
        const ratio = targetParams / currentParams
        if (Math.abs(currentParams - targetParams) / targetParams < 0.005) {
          break
        }
        // Scale each dFf and snap to multiple of 64
        dFfs = dFfs.map(ff => Math.max(512, Math.round((ff * ratio) / 64) * 64))
      }
    }

    return buildConfig(dFfs)
  }

  /**
   * Computes exact parameter breakdown and theoretical FLOPs per token for any geometry.
   */
  public static calculateParameters(config: ModelArchitectureConfig): ParameterBreakdown {
    const embedDim = 768
    const tokenEmbedParams = config.vocabSize * embedDim
    const posEmbedParams = config.maxSeqLen * embedDim
    const embeddingParams = tokenEmbedParams + posEmbedParams

    let attentionParams = 0
    let mlpParams = 0
    let projectionParams = 0
    let normParams = 0

    let lastDim = embedDim

    for (const l of config.layers) {
      // Linear projection if dim changes
      if (l.dModelIn !== l.dModelOut) {
        projectionParams += l.dModelIn * l.dModelOut
      }

      // Attention params: Q, K, V, O projections (4 * dModelOut * dModelOut)
      attentionParams += 4 * l.dModelOut * l.dModelOut

      // LayerNorm params (norm1 and norm2: 2 * dModelOut each for gamma + beta)
      normParams += 4 * l.dModelOut

      // MLP params: fc1 (dModelOut * dFf) + fc2 (dFf * dModelOut)
      mlpParams += 2 * l.dModelOut * l.dFf

      lastDim = l.dModelOut
    }

    // Final LayerNorm: gamma + beta
    normParams += 2 * lastDim

    // LM Head: lastDim * vocabSize
    const headParams = lastDim * config.vocabSize

    const totalParams =
      embeddingParams +
      attentionParams +
      mlpParams +
      projectionParams +
      normParams +
      headParams

    // Theoretical FLOPs per token in forward pass is approximately 2 * totalParams
    const theoreticalFlopsPerToken = 2 * totalParams

    return {
      embeddingParams,
      attentionParams,
      mlpParams,
      projectionParams,
      normParams,
      headParams,
      totalParams,
      theoreticalFlopsPerToken
    }
  }

  /**
   * Validates IsoFLOP parity between a candidate model and baseline.
   * Returns delta percentage and boolean indicating if within tolerance.
   */
  public static verifyIsoFlopParity(
    baseline: ParameterBreakdown,
    candidate: ParameterBreakdown,
    tolerancePercent: number = 2.0
  ): { deltaPercent: number; isWithinParity: boolean } {
    const delta = ((candidate.totalParams - baseline.totalParams) / baseline.totalParams) * 100.0
    return {
      deltaPercent: delta,
      isWithinParity: Math.abs(delta) <= tolerancePercent
    }
  }
}
