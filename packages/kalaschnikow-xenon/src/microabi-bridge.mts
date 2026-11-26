/**
 * @module
 * B8G Microkernel ABI Bridge (`microabi-bridge.mts`).
 *
 * Provides zero-allocation linear memory dispatch for neural computation kernels:
 *   - `OP_GEMM` (General Matrix-Vector Multiplication)
 *   - `OP_BLOCK_SPARSE_GEMM` (Block-sparse linear kernel skipping inactive blocks)
 *   - `OP_LAYER_NORM` (In-place Layer Normalization)
 *   - `OP_GELU` (Fast vectorized GELU activation)
 *   - `OP_SOFTMAX` (Numerically stable in-place Softmax)
 *   - `OP_RESIDUAL_ADD` (Element-wise residual addition)
 *
 * Operates directly on SharedArrayBuffer/Float32Array linear views with zero temporary garbage.
 */

export const UABI_OP_GEMM = 0x50
export const UABI_OP_BLOCK_SPARSE_GEMM = 0x51
export const UABI_OP_LAYER_NORM = 0x52
export const UABI_OP_GELU = 0x53
export const UABI_OP_SOFTMAX = 0x54
export const UABI_OP_RESIDUAL_ADD = 0x55

export class B8GMicroABIBridge {
  private static dispatchCount = 0

  public static getDispatchCount(): number {
    return this.dispatchCount
  }

  /**
   * Linear Matrix Multiplication: out = inVec * weight^T + bias
   * inVec: shape [inFeatures]
   * weight: shape [outFeatures, inFeatures] (row-major)
   * bias: optional shape [outFeatures]
   * out: shape [outFeatures]
   */
  public static gemm(
    inVec: Float32Array,
    weight: Float32Array,
    out: Float32Array,
    inFeatures: number,
    outFeatures: number,
    bias?: Float32Array
  ): void {
    this.dispatchCount++
    for (let r = 0; r < outFeatures; r++) {
      let sum = bias ? bias[r] : 0.0
      const rowOffset = r * inFeatures
      // Inner loop vector dot-product
      for (let c = 0; c < inFeatures; c++) {
        sum += inVec[c] * weight[rowOffset + c]
      }
      out[r] = sum
    }
  }

  /**
   * Block-Sparse GEMM: skips entire block_size x block_size blocks marked false in the block mask.
   * blockMask: Uint8Array of shape [outFeatures / blockSize, inFeatures / blockSize]
   */
  public static blockSparseGemm(
    inVec: Float32Array,
    weight: Float32Array,
    out: Float32Array,
    inFeatures: number,
    outFeatures: number,
    blockSize: number,
    blockMask: Uint8Array,
    bias?: Float32Array
  ): void {
    this.dispatchCount++
    const numBlocksH = outFeatures / blockSize
    const numBlocksW = inFeatures / blockSize

    // Initialize output with bias or zero
    if (bias) {
      out.set(bias)
    } else {
      out.fill(0)
    }

    for (let bh = 0; bh < numBlocksH; bh++) {
      const outRowBase = bh * blockSize
      const maskRowOffset = bh * numBlocksW

      for (let bw = 0; bw < numBlocksW; bw++) {
        // Skip inactive block!
        if (blockMask[maskRowOffset + bw] === 0) {
          continue
        }

        const inColBase = bw * blockSize
        // Execute active block sub-GEMM
        for (let r = 0; r < blockSize; r++) {
          const outIdx = outRowBase + r
          const weightRowOffset = (outRowBase + r) * inFeatures + inColBase
          let sum = 0.0
          for (let c = 0; c < blockSize; c++) {
            sum += inVec[inColBase + c] * weight[weightRowOffset + c]
          }
          out[outIdx] += sum
        }
      }
    }
  }

  /**
   * Layer Normalization: out = (x - mean) / sqrt(var + eps) * gamma + beta
   */
  public static layerNorm(
    x: Float32Array,
    out: Float32Array,
    gamma: Float32Array,
    beta: Float32Array,
    dim: number,
    eps: number = 1e-5
  ): void {
    this.dispatchCount++
    let sum = 0.0
    for (let i = 0; i < dim; i++) {
      sum += x[i]
    }
    const mean = sum / dim

    let sumSqDiff = 0.0
    for (let i = 0; i < dim; i++) {
      const diff = x[i] - mean
      sumSqDiff += diff * diff
    }
    const variance = sumSqDiff / dim
    const invStd = 1.0 / Math.sqrt(variance + eps)

    for (let i = 0; i < dim; i++) {
      out[i] = ((x[i] - mean) * invStd) * gamma[i] + beta[i]
    }
  }

  /**
   * Vectorized GELU activation (Hendrycks & Gimpel approximation):
   * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
   */
  public static gelu(x: Float32Array, out: Float32Array, len: number): void {
    this.dispatchCount++
    const SQRT_2_OVER_PI = 0.7978845608028654
    for (let i = 0; i < len; i++) {
      const v = x[i]
      const cube = v * v * v
      const inner = SQRT_2_OVER_PI * (v + 0.044715 * cube)
      out[i] = 0.5 * v * (1.0 + Math.tanh(inner))
    }
  }

  /**
   * In-place Softmax over vector:
   */
  public static softmax(vec: Float32Array, len: number): void {
    this.dispatchCount++
    let maxVal = -Infinity
    for (let i = 0; i < len; i++) {
      if (vec[i] > maxVal) maxVal = vec[i]
    }

    let sumExp = 0.0
    for (let i = 0; i < len; i++) {
      const e = Math.exp(vec[i] - maxVal)
      vec[i] = e
      sumExp += e
    }

    const invSum = 1.0 / sumExp
    for (let i = 0; i < len; i++) {
      vec[i] *= invSum
    }
  }

  /**
   * In-place residual addition: target[i] += source[i]
   */
  public static residualAdd(target: Float32Array, source: Float32Array, len: number): void {
    this.dispatchCount++
    for (let i = 0; i < len; i++) {
      target[i] += source[i]
    }
  }
}
