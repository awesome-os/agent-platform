/**
 * Spin-Wave Phasor Mixture-of-Experts Model (`SpinWaveMoEModel`)
 *
 * Trainable, deterministic, INT8-quantizable sparse MoE classifier using
 * complex unit-circle phasor embeddings and magnonic phase-coherence Top-K
 * routing. Zero external dependencies, erasable TypeScript syntax only.
 *
 * Binary `.b8p` Format (`B8SW` v1 / `B8G_ABI_MAGIC_V3 = 0x33564742`):
 *   Header [bytes 0..63] (64-byte aligned, little-endian):
 *     0..3   : ASCII magic "B8SW" (0x42, 0x38, 0x53, 0x57)
 *     4..7   : uint32 version = 1
 *     8..11  : uint32 abiMagic = 0x33564742 (B8G_ABI_MAGIC_V3)
 *     12..15 : uint32 vocabSize
 *     16..19 : uint32 dPhasor
 *     20..23 : uint32 numExperts
 *     24..27 : uint32 topK
 *     28..31 : uint32 numClasses
 *     32..35 : float32 scaleEncRe
 *     36..39 : float32 scaleEncIm
 *     40..43 : float32 scaleExpRe
 *     44..47 : float32 scaleExpIm
 *     48..51 : float32 scaleHeadWRe
 *     52..55 : float32 scaleHeadWIm
 *     56..59 : float32 scaleHeadBias
 *     60..63 : uint32 labelsByteLength
 *   Payload [bytes 64..end-4]:
 *     INT8 tensors in order: encRe, encIm, expRe, expIm, headWRe, headWIm, headBias
 *     UTF-8 class labels (newline-delimited, labelsByteLength bytes)
 *   Trailer [last 4 bytes]:
 *     uint32 FNV-1a 32-bit checksum over bytes [0..end-4)
 */

export const B8SW_MAGIC = 'B8SW';
export const B8SW_VERSION = 1;
export const B8G_ABI_MAGIC_V3 = 0x33564742;
export const B8SW_HEADER_BYTES = 64;
export const MAX_EXPERTS = 64;

const FNV_OFFSET_BASIS = 0x811c9dc5;
const FNV_PRIME = 0x01000193;
const NORM_EPS = 1e-8;
const LOG_EPS = 1e-12;

export type FeatureInput = string | readonly string[];

export interface SpinWaveMoEConfig {
  readonly vocabSize?: number;
  readonly dPhasor?: number;
  readonly numExperts?: number;
  readonly topK?: number;
  readonly numClasses?: number;
  readonly classLabels?: readonly string[];
  readonly seed?: number;
}

export interface TrainingSample {
  readonly input: FeatureInput;
  readonly targetClass: number;
}

export interface TrainOptions {
  readonly epochs?: number;
  readonly learningRate?: number;
  readonly lrDecay?: number;
  readonly weightDecay?: number;
  readonly seed?: number;
  readonly shuffle?: boolean;
}

export interface TrainMetrics {
  readonly epochs: number;
  readonly finalLoss: number;
  readonly trainAccuracy: number;
  readonly lossHistory: readonly number[];
}

export interface SpinWavePrediction {
  readonly classIndex: number;
  readonly classLabel: string;
  readonly confidence: number;
  readonly probabilities: Float32Array;
  readonly logits: Float32Array;
  readonly activeExperts: readonly number[];
  readonly gateWeights: Float32Array;
  readonly expertCoherences: Float32Array;
  readonly phasorRe: Float32Array;
  readonly phasorIm: Float32Array;
}

export interface CrystallizeResult {
  readonly beforeClass: number;
  readonly afterClass: number;
  readonly beforeConfidence: number;
  readonly afterConfidence: number;
  readonly stepsRun: number;
  readonly finalLoss: number;
}

export interface EvaluationMetrics {
  readonly accuracy: number;
  readonly correct: number;
  readonly total: number;
  readonly meanLoss: number;
}

export interface B8swHeaderInfo {
  readonly magic: string;
  readonly version: number;
  readonly abiMagic: number;
  readonly vocabSize: number;
  readonly dPhasor: number;
  readonly numExperts: number;
  readonly topK: number;
  readonly numClasses: number;
  readonly scales: {
    readonly encRe: number;
    readonly encIm: number;
    readonly expRe: number;
    readonly expIm: number;
    readonly headWRe: number;
    readonly headWIm: number;
    readonly headBias: number;
  };
  readonly labelsByteLength: number;
  readonly totalBytes: number;
}

interface SparseFeatures {
  readonly indices: Uint16Array;
  readonly values: Float32Array;
  readonly count: number;
}

/**
 * Deterministic Mulberry32 PRNG returning values in [0, 1).
 */
export function createMulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * 32-bit FNV-1a hash over a UTF-16/ASCII string.
 */
export function fnv1a32String(text: string): number {
  let hash = FNV_OFFSET_BASIS;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i) & 0xff;
    hash = Math.imul(hash, FNV_PRIME) >>> 0;
  }
  return hash >>> 0;
}

/**
 * 32-bit FNV-1a hash over a byte slice.
 */
export function fnv1a32Bytes(bytes: Uint8Array, length: number = bytes.byteLength): number {
  let hash = FNV_OFFSET_BASIS;
  for (let i = 0; i < length; i++) {
    hash ^= bytes[i]!;
    hash = Math.imul(hash, FNV_PRIME) >>> 0;
  }
  return hash >>> 0;
}

function quantizeInt8Symmetric(src: Float32Array, dst: Int8Array, offset: number): number {
  let maxAbs = 0;
  for (let i = 0; i < src.length; i++) {
    const a = Math.abs(src[i]!);
    if (a > maxAbs) maxAbs = a;
  }
  const scale = maxAbs > 0 ? maxAbs / 127 : 1;
  const invScale = maxAbs > 0 ? 127 / maxAbs : 0;
  for (let i = 0; i < src.length; i++) {
    const q = Math.round(src[i]! * invScale);
    dst[offset + i] = q > 127 ? 127 : q < -127 ? -127 : q;
  }
  return scale;
}

function dequantizeInt8Symmetric(
  src: Int8Array,
  offset: number,
  length: number,
  scale: number,
  dst: Float32Array,
): void {
  for (let i = 0; i < length; i++) {
    dst[i] = src[offset + i]! * scale;
  }
}

/**
 * Parse and validate the 64-byte `B8SW` header from a `.b8p` buffer.
 */
export function parseB8pHeader(buffer: Uint8Array): B8swHeaderInfo {
  if (buffer.byteLength < B8SW_HEADER_BYTES + 4) {
    throw new RangeError(
      `Buffer too small for B8SW .b8p header: ${buffer.byteLength} < ${B8SW_HEADER_BYTES + 4}`,
    );
  }

  const magic = String.fromCharCode(buffer[0]!, buffer[1]!, buffer[2]!, buffer[3]!);
  if (magic !== B8SW_MAGIC) {
    throw new Error(`Invalid B8SW magic: expected "${B8SW_MAGIC}", got "${magic}"`);
  }

  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const version = view.getUint32(4, true);
  if (version !== B8SW_VERSION) {
    throw new Error(`Unsupported B8SW version: expected ${B8SW_VERSION}, got ${version}`);
  }

  const abiMagic = view.getUint32(8, true);
  if (abiMagic !== B8G_ABI_MAGIC_V3) {
    throw new Error(
      `Invalid B8G_ABI_MAGIC_V3: expected 0x${B8G_ABI_MAGIC_V3.toString(16)}, got 0x${abiMagic.toString(16)}`,
    );
  }

  const vocabSize = view.getUint32(12, true);
  const dPhasor = view.getUint32(16, true);
  const numExperts = view.getUint32(20, true);
  const topK = view.getUint32(24, true);
  const numClasses = view.getUint32(28, true);

  const encRe = view.getFloat32(32, true);
  const encIm = view.getFloat32(36, true);
  const expRe = view.getFloat32(40, true);
  const expIm = view.getFloat32(44, true);
  const headWRe = view.getFloat32(48, true);
  const headWIm = view.getFloat32(52, true);
  const headBias = view.getFloat32(56, true);
  const labelsByteLength = view.getUint32(60, true);

  return {
    magic,
    version,
    abiMagic,
    vocabSize,
    dPhasor,
    numExperts,
    topK,
    numClasses,
    scales: {
      encRe,
      encIm,
      expRe,
      expIm,
      headWRe,
      headWIm,
      headBias,
    },
    labelsByteLength,
    totalBytes: buffer.byteLength,
  };
}

/**
 * Trainable, INT8-quantizable Spin-Wave Phasor Mixture-of-Experts Model.
 */
export class SpinWaveMoEModel {
  readonly vocabSize: number;
  readonly dPhasor: number;
  readonly numExperts: number;
  readonly topK: number;
  readonly numClasses: number;
  readonly classLabels: readonly string[];

  // Complex phasor encoder: [vocabSize, dPhasor]
  readonly encRe: Float32Array;
  readonly encIm: Float32Array;

  // Unit-circle complex expert phase keys: [numExperts, dPhasor]
  readonly expRe: Float32Array;
  readonly expIm: Float32Array;

  // Sparse expert classification heads: [numExperts, numClasses, dPhasor] + [numExperts, numClasses]
  readonly headWRe: Float32Array;
  readonly headWIm: Float32Array;
  readonly headBias: Float32Array;

  readonly #routerScale: number;

  // Reusable scratch buffers for zero-allocation hot-path forward/backward
  readonly #bucketScratch: Float32Array;
  readonly #activeBucketIdx: Uint16Array;
  readonly #activeBucketVal: Float32Array;
  readonly #rawRe: Float32Array;
  readonly #rawIm: Float32Array;
  readonly #mag: Float32Array;
  readonly #qRe: Float32Array;
  readonly #qIm: Float32Array;
  readonly #coherences: Float32Array;
  readonly #topIdx: Int32Array;
  readonly #topScore: Float32Array;
  readonly #gate: Float32Array;
  readonly #expertLogits: Float32Array;
  readonly #logits: Float32Array;
  readonly #probs: Float32Array;
  readonly #dLogits: Float32Array;
  readonly #dGate: Float32Array;
  readonly #gqRe: Float32Array;
  readonly #gqIm: Float32Array;
  readonly #gRawRe: Float32Array;
  readonly #gRawIm: Float32Array;

  constructor(config: SpinWaveMoEConfig = {}) {
    const vocabSize = config.vocabSize ?? 256;
    const dPhasor = config.dPhasor ?? 16;
    const numExperts = config.numExperts ?? 16;
    const topK = config.topK ?? 2;
    const numClasses = config.classLabels ? config.classLabels.length : (config.numClasses ?? 4);
    const seed = config.seed ?? 0x51a7f00d;

    if (!Number.isInteger(vocabSize) || vocabSize < 8 || vocabSize > 65535) {
      throw new RangeError(`vocabSize must be an integer in [8, 65535], got ${vocabSize}`);
    }
    if (!Number.isInteger(dPhasor) || dPhasor < 2 || dPhasor > 256) {
      throw new RangeError(`dPhasor must be an integer in [2, 256], got ${dPhasor}`);
    }
    if (!Number.isInteger(numExperts) || numExperts < 1 || numExperts > MAX_EXPERTS) {
      throw new RangeError(
        `numExperts must be an integer in [1, ${MAX_EXPERTS}], got ${numExperts}`,
      );
    }
    if (!Number.isInteger(topK) || topK < 1 || topK > numExperts) {
      throw new RangeError(`topK must be an integer in [1, ${numExperts}], got ${topK}`);
    }
    if (!Number.isInteger(numClasses) || numClasses < 2 || numClasses > 256) {
      throw new RangeError(`numClasses must be an integer in [2, 256], got ${numClasses}`);
    }

    this.vocabSize = vocabSize;
    this.dPhasor = dPhasor;
    this.numExperts = numExperts;
    this.topK = topK;
    this.numClasses = numClasses;
    this.classLabels = config.classLabels
      ? [...config.classLabels]
      : Array.from({ length: numClasses }, (_, i) => `class-${i}`);

    if (this.classLabels.length !== numClasses) {
      throw new RangeError(
        `classLabels length (${this.classLabels.length}) must match numClasses (${numClasses})`,
      );
    }

    this.#routerScale = 1 / Math.sqrt(dPhasor);

    this.encRe = new Float32Array(vocabSize * dPhasor);
    this.encIm = new Float32Array(vocabSize * dPhasor);
    this.expRe = new Float32Array(numExperts * dPhasor);
    this.expIm = new Float32Array(numExperts * dPhasor);
    this.headWRe = new Float32Array(numExperts * numClasses * dPhasor);
    this.headWIm = new Float32Array(numExperts * numClasses * dPhasor);
    this.headBias = new Float32Array(numExperts * numClasses);

    // Scratch buffers
    this.#bucketScratch = new Float32Array(vocabSize);
    this.#activeBucketIdx = new Uint16Array(vocabSize);
    this.#activeBucketVal = new Float32Array(vocabSize);
    this.#rawRe = new Float32Array(dPhasor);
    this.#rawIm = new Float32Array(dPhasor);
    this.#mag = new Float32Array(dPhasor);
    this.#qRe = new Float32Array(dPhasor);
    this.#qIm = new Float32Array(dPhasor);
    this.#coherences = new Float32Array(numExperts);
    this.#topIdx = new Int32Array(topK);
    this.#topScore = new Float32Array(topK);
    this.#gate = new Float32Array(topK);
    this.#expertLogits = new Float32Array(topK * numClasses);
    this.#logits = new Float32Array(numClasses);
    this.#probs = new Float32Array(numClasses);
    this.#dLogits = new Float32Array(numClasses);
    this.#dGate = new Float32Array(topK);
    this.#gqRe = new Float32Array(dPhasor);
    this.#gqIm = new Float32Array(dPhasor);
    this.#gRawRe = new Float32Array(dPhasor);
    this.#gRawIm = new Float32Array(dPhasor);

    this.#initWeights(seed);
  }

  #initWeights(seed: number): void {
    const rng = createMulberry32(seed);
    const encScale = 0.6 / Math.sqrt(this.dPhasor);
    const headScale = 0.25 / Math.sqrt(2 * this.dPhasor);

    for (let i = 0; i < this.encRe.length; i++) {
      this.encRe[i] = (rng() * 2 - 1) * encScale;
      this.encIm[i] = (rng() * 2 - 1) * encScale;
    }

    // Initialize expert keys uniformly on the complex unit circle (re^2 + im^2 = 1)
    for (let i = 0; i < this.expRe.length; i++) {
      const theta = rng() * Math.PI * 2;
      this.expRe[i] = Math.cos(theta);
      this.expIm[i] = Math.sin(theta);
    }

    for (let i = 0; i < this.headWRe.length; i++) {
      this.headWRe[i] = (rng() * 2 - 1) * headScale;
      this.headWIm[i] = (rng() * 2 - 1) * headScale;
    }
    this.headBias.fill(0);
  }

  /**
   * Hash text or token array into L2-normalized sparse FNV-1a feature buckets.
   */
  #extractSparseFeatures(input: FeatureInput): SparseFeatures {
    const buckets = this.#bucketScratch;
    const activeIdx = this.#activeBucketIdx;
    const activeVal = this.#activeBucketVal;
    const V = this.vocabSize;
    let count = 0;

    const addFeature = (token: string, weight: number): void => {
      const idx = fnv1a32String(token) % V;
      if (buckets[idx] === 0) {
        activeIdx[count++] = idx;
      }
      buckets[idx] = buckets[idx]! + weight;
    };

    const rawTokens: string[] = [];
    if (typeof input === 'string') {
      const lower = input.toLowerCase();
      let start = -1;
      for (let i = 0; i <= lower.length; i++) {
        const ch = i < lower.length ? lower.charCodeAt(i) : 0;
        const isAlnum =
          (ch >= 97 && ch <= 122) || (ch >= 48 && ch <= 57) || ch === 95 || ch === 45;
        if (isAlnum) {
          if (start === -1) start = i;
        } else if (start !== -1) {
          rawTokens.push(lower.slice(start, i));
          start = -1;
        }
      }
    } else {
      for (let i = 0; i < input.length; i++) {
        const t = input[i]!.toLowerCase().trim();
        if (t.length > 0) rawTokens.push(t);
      }
    }

    for (let i = 0; i < rawTokens.length; i++) {
      const tok = rawTokens[i]!;
      // Whole token feature (strongest signal)
      addFeature(`w:${tok}`, 2.0);

      // Sub-tokens split on '-' or '_' for compound identifiers (e.g. cas-storage, b8g-build)
      if (tok.includes('-') || tok.includes('_')) {
        const parts = tok.split(/[-_]+/);
        for (let p = 0; p < parts.length; p++) {
          if (parts[p]!.length >= 2) {
            addFeature(`s:${parts[p]!}`, 1.25);
          }
        }
      }

      // Character 3-grams for morphological generalization on unseen words
      if (tok.length >= 3) {
        const padded = `^${tok}$`;
        for (let g = 0; g <= padded.length - 3; g++) {
          addFeature(`g:${padded.slice(g, g + 3)}`, 0.45);
        }
      }

      // Token bigrams for phrase context
      if (i + 1 < rawTokens.length) {
        addFeature(`b:${tok}_${rawTokens[i + 1]!}`, 1.0);
      }
    }

    // Fallback bucket for empty input
    if (count === 0) {
      activeIdx[0] = 0;
      buckets[0] = 1.0;
      count = 1;
    }

    // L2-normalize active buckets and clear scratch for next call
    let normSq = 0;
    for (let i = 0; i < count; i++) {
      const idx = activeIdx[i]!;
      const v = buckets[idx]!;
      normSq += v * v;
    }
    const invNorm = 1 / Math.sqrt(normSq + NORM_EPS);
    for (let i = 0; i < count; i++) {
      const idx = activeIdx[i]!;
      activeVal[i] = buckets[idx]! * invNorm;
      buckets[idx] = 0;
    }

    return { indices: activeIdx, values: activeVal, count };
  }

  /**
   * Internal zero-allocation forward pass populating scratch buffers.
   * Returns predicted class index.
   */
  #forward(features: SparseFeatures): number {
    const D = this.dPhasor;
    const E = this.numExperts;
    const K = this.topK;
    const C = this.numClasses;

    const rawRe = this.#rawRe;
    const rawIm = this.#rawIm;
    const mag = this.#mag;
    const qRe = this.#qRe;
    const qIm = this.#qIm;

    rawRe.fill(0);
    rawIm.fill(0);

    // 1. Phasor Encoder: sparse FNV-1a features -> complex unit-circle phasor (qRe, qIm)
    const fIdx = features.indices;
    const fVal = features.values;
    const fCount = features.count;
    const encRe = this.encRe;
    const encIm = this.encIm;

    for (let f = 0; f < fCount; f++) {
      const base = fIdx[f]! * D;
      const xv = fVal[f]!;
      for (let d = 0; d < D; d++) {
        rawRe[d] = rawRe[d]! + xv * encRe[base + d]!;
        rawIm[d] = rawIm[d]! + xv * encIm[base + d]!;
      }
    }

    for (let d = 0; d < D; d++) {
      const r = rawRe[d]!;
      const im = rawIm[d]!;
      const m = Math.sqrt(r * r + im * im + NORM_EPS);
      mag[d] = m;
      qRe[d] = r / m;
      qIm[d] = im / m;
    }

    // 2. Magnonic Phase-Coherence Top-K Router (pure FMA dot product)
    const expRe = this.expRe;
    const expIm = this.expIm;
    const coherences = this.#coherences;
    const topIdx = this.#topIdx;
    const topScore = this.#topScore;

    for (let i = 0; i < K; i++) {
      topIdx[i] = -1;
      topScore[i] = -Infinity;
    }

    for (let k = 0; k < E; k++) {
      const base = k * D;
      let coh = 0;
      for (let d = 0; d < D; d++) {
        coh += qRe[d]! * expRe[base + d]! + qIm[d]! * expIm[base + d]!;
      }
      coherences[k] = coh;

      // Insertion into sorted Top-K buffer (descending)
      if (coh > topScore[K - 1]!) {
        let pos = K - 1;
        while (pos > 0 && coh > topScore[pos - 1]!) {
          topScore[pos] = topScore[pos - 1]!;
          topIdx[pos] = topIdx[pos - 1]!;
          pos--;
        }
        topScore[pos] = coh;
        topIdx[pos] = k;
      }
    }

    // Softmax over Top-K scaled coherences
    const gate = this.#gate;
    const scale = this.#routerScale;
    const maxTop = topScore[0]! * scale;
    let gateSum = 0;
    for (let i = 0; i < K; i++) {
      const g = Math.exp(topScore[i]! * scale - maxTop);
      gate[i] = g;
      gateSum += g;
    }
    const invGateSum = 1 / gateSum;
    for (let i = 0; i < K; i++) {
      gate[i] = gate[i]! * invGateSum;
    }

    // 3. Sparse Expert Heads: evaluate ONLY the Top-K active experts
    const headWRe = this.headWRe;
    const headWIm = this.headWIm;
    const headBias = this.headBias;
    const expertLogits = this.#expertLogits;
    const logits = this.#logits;
    logits.fill(0);

    for (let i = 0; i < K; i++) {
      const k = topIdx[i]!;
      const g = gate[i]!;
      const expertHeadBase = k * C;
      const slotOffset = i * C;

      for (let c = 0; c < C; c++) {
        const wBase = (expertHeadBase + c) * D;
        let acc = headBias[expertHeadBase + c]!;
        for (let d = 0; d < D; d++) {
          acc += qRe[d]! * headWRe[wBase + d]! + qIm[d]! * headWIm[wBase + d]!;
        }
        expertLogits[slotOffset + c] = acc;
        logits[c] = logits[c]! + g * acc;
      }
    }

    // 4. Class softmax & argmax
    let maxLogit = logits[0]!;
    let bestClass = 0;
    for (let c = 1; c < C; c++) {
      if (logits[c]! > maxLogit) {
        maxLogit = logits[c]!;
        bestClass = c;
      }
    }

    const probs = this.#probs;
    let probSum = 0;
    for (let c = 0; c < C; c++) {
      const e = Math.exp(logits[c]! - maxLogit);
      probs[c] = e;
      probSum += e;
    }
    const invProbSum = 1 / probSum;
    for (let c = 0; c < C; c++) {
      probs[c] = probs[c]! * invProbSum;
    }

    return bestClass;
  }

  /**
   * Single-sample exact SGD backward pass with unit-circle phase re-normalization.
   * Returns cross-entropy loss for this sample.
   */
  #trainStep(
    features: SparseFeatures,
    targetClass: number,
    lr: number,
    weightDecay: number,
  ): number {
    this.#forward(features);

    const D = this.dPhasor;
    const K = this.topK;
    const C = this.numClasses;
    const probs = this.#probs;
    const loss = -Math.log(Math.max(probs[targetClass]!, LOG_EPS));

    const dLogits = this.#dLogits;
    for (let c = 0; c < C; c++) {
      dLogits[c] = probs[c]! - (c === targetClass ? 1 : 0);
    }

    const topIdx = this.#topIdx;
    const gate = this.#gate;
    const expertLogits = this.#expertLogits;
    const dGate = this.#dGate;
    const gqRe = this.#gqRe;
    const gqIm = this.#gqIm;
    gqRe.fill(0);
    gqIm.fill(0);

    const qRe = this.#qRe;
    const qIm = this.#qIm;
    const headWRe = this.headWRe;
    const headWIm = this.headWIm;
    const headBias = this.headBias;

    // 1. Backprop through active sparse expert heads
    for (let i = 0; i < K; i++) {
      const k = topIdx[i]!;
      const g = gate[i]!;
      const expertHeadBase = k * C;
      const slotOffset = i * C;
      let dg = 0;

      for (let c = 0; c < C; c++) {
        const dL = dLogits[c]!;
        dg += dL * expertLogits[slotOffset + c]!;
        const dExpLogit = g * dL;
        const wBase = (expertHeadBase + c) * D;

        // Accumulate phasor gradient before mutating head weights
        for (let d = 0; d < D; d++) {
          const wr = headWRe[wBase + d]!;
          const wi = headWIm[wBase + d]!;
          gqRe[d] = gqRe[d]! + dExpLogit * wr;
          gqIm[d] = gqIm[d]! + dExpLogit * wi;

          headWRe[wBase + d] = wr - lr * (dExpLogit * qRe[d]! + weightDecay * wr);
          headWIm[wBase + d] = wi - lr * (dExpLogit * qIm[d]! + weightDecay * wi);
        }
        headBias[expertHeadBase + c] = headBias[expertHeadBase + c]! - lr * dExpLogit;
      }
      dGate[i] = dg;
    }

    // 2. Backprop through Top-K gate softmax & update active expert phase keys
    let dotGate = 0;
    for (let i = 0; i < K; i++) {
      dotGate += gate[i]! * dGate[i]!;
    }

    const expRe = this.expRe;
    const expIm = this.expIm;
    const routerScale = this.#routerScale;

    for (let i = 0; i < K; i++) {
      const k = topIdx[i]!;
      const dCoh = routerScale * gate[i]! * (dGate[i]! - dotGate);
      const base = k * D;

      for (let d = 0; d < D; d++) {
        const er = expRe[base + d]!;
        const ei = expIm[base + d]!;
        gqRe[d] = gqRe[d]! + dCoh * er;
        gqIm[d] = gqIm[d]! + dCoh * ei;

        // Update expert phase key and re-project onto the unit circle (re^2 + im^2 = 1)
        const nr = er - lr * dCoh * qRe[d]!;
        const ni = ei - lr * dCoh * qIm[d]!;
        const invNorm = 1 / Math.sqrt(nr * nr + ni * ni + NORM_EPS);
        expRe[base + d] = nr * invNorm;
        expIm[base + d] = ni * invNorm;
      }
    }

    // 3. Backprop through unit-circle phasor normalization: q = raw / ||raw||
    const mag = this.#mag;
    const gRawRe = this.#gRawRe;
    const gRawIm = this.#gRawIm;

    for (let d = 0; d < D; d++) {
      const gr = gqRe[d]!;
      const gi = gqIm[d]!;
      const qr = qRe[d]!;
      const qi = qIm[d]!;
      const dotQ = gr * qr + gi * qi;
      const invMag = 1 / mag[d]!;
      gRawRe[d] = (gr - dotQ * qr) * invMag;
      gRawIm[d] = (gi - dotQ * qi) * invMag;
    }

    // 4. Update phasor encoder weights for active hashed feature buckets
    const encRe = this.encRe;
    const encIm = this.encIm;
    const fIdx = features.indices;
    const fVal = features.values;
    const fCount = features.count;

    for (let f = 0; f < fCount; f++) {
      const base = fIdx[f]! * D;
      const xv = fVal[f]!;
      for (let d = 0; d < D; d++) {
        const er = encRe[base + d]!;
        const ei = encIm[base + d]!;
        encRe[base + d] = er - lr * (gRawRe[d]! * xv + weightDecay * er);
        encIm[base + d] = ei - lr * (gRawIm[d]! * xv + weightDecay * ei);
      }
    }

    return loss;
  }

  /**
   * Run inference on an input string or token array.
   */
  predict(input: FeatureInput): SpinWavePrediction {
    const features = this.#extractSparseFeatures(input);
    const classIndex = this.#forward(features);

    const activeExperts: number[] = [];
    for (let i = 0; i < this.topK; i++) {
      activeExperts.push(this.#topIdx[i]!);
    }

    return {
      classIndex,
      classLabel: this.classLabels[classIndex]!,
      confidence: this.#probs[classIndex]!,
      probabilities: new Float32Array(this.#probs),
      logits: new Float32Array(this.#logits),
      activeExperts,
      gateWeights: new Float32Array(this.#gate),
      expertCoherences: new Float32Array(this.#coherences),
      phasorRe: new Float32Array(this.#qRe),
      phasorIm: new Float32Array(this.#qIm),
    };
  }

  /**
   * Fast zero-allocation classification returning only the predicted class index.
   */
  predictClass(input: FeatureInput): number {
    const features = this.#extractSparseFeatures(input);
    return this.#forward(features);
  }

  /**
   * Multi-epoch deterministic SGD training with Mulberry32 shuffling and
   * unit-circle phase normalization.
   */
  train(samples: readonly TrainingSample[], options: TrainOptions = {}): TrainMetrics {
    if (samples.length === 0) {
      throw new RangeError('Training dataset must contain at least one sample');
    }

    const epochs = options.epochs ?? 40;
    const initialLr = options.learningRate ?? 0.18;
    const lrDecay = options.lrDecay ?? 0.97;
    const weightDecay = options.weightDecay ?? 1e-4;
    const shuffle = options.shuffle ?? true;
    const rng = createMulberry32(options.seed ?? 0x1337b857);

    // Pre-extract sparse features once for fast deterministic epochs
    const cachedIndices: Uint16Array[] = new Array(samples.length);
    const cachedValues: Float32Array[] = new Array(samples.length);
    const targets = new Int32Array(samples.length);

    for (let i = 0; i < samples.length; i++) {
      const sample = samples[i]!;
      if (
        !Number.isInteger(sample.targetClass) ||
        sample.targetClass < 0 ||
        sample.targetClass >= this.numClasses
      ) {
        throw new RangeError(
          `Sample ${i} targetClass ${sample.targetClass} out of range [0, ${this.numClasses - 1}]`,
        );
      }
      const feat = this.#extractSparseFeatures(sample.input);
      cachedIndices[i] = feat.indices.slice(0, feat.count);
      cachedValues[i] = feat.values.slice(0, feat.count);
      targets[i] = sample.targetClass;
    }

    const order = new Int32Array(samples.length);
    for (let i = 0; i < samples.length; i++) order[i] = i;

    const lossHistory: number[] = [];
    let lr = initialLr;

    for (let epoch = 0; epoch < epochs; epoch++) {
      if (shuffle) {
        for (let i = order.length - 1; i > 0; i--) {
          const j = Math.floor(rng() * (i + 1));
          const tmp = order[i]!;
          order[i] = order[j]!;
          order[j] = tmp;
        }
      }

      let epochLoss = 0;
      for (let i = 0; i < order.length; i++) {
        const idx = order[i]!;
        const feat: SparseFeatures = {
          indices: cachedIndices[idx]!,
          values: cachedValues[idx]!,
          count: cachedIndices[idx]!.length,
        };
        epochLoss += this.#trainStep(feat, targets[idx]!, lr, weightDecay);
      }

      lossHistory.push(epochLoss / samples.length);
      lr *= lrDecay;
    }

    let correct = 0;
    for (let i = 0; i < samples.length; i++) {
      const feat: SparseFeatures = {
        indices: cachedIndices[i]!,
        values: cachedValues[i]!,
        count: cachedIndices[i]!.length,
      };
      if (this.#forward(feat) === targets[i]!) correct++;
    }

    return {
      epochs,
      finalLoss: lossHistory[lossHistory.length - 1] ?? 0,
      trainAccuracy: correct / samples.length,
      lossHistory,
    };
  }

  /**
   * Evaluate accuracy and mean cross-entropy loss on a dataset.
   */
  evaluate(samples: readonly TrainingSample[]): EvaluationMetrics {
    if (samples.length === 0) {
      return { accuracy: 0, correct: 0, total: 0, meanLoss: 0 };
    }
    let correct = 0;
    let totalLoss = 0;
    for (let i = 0; i < samples.length; i++) {
      const s = samples[i]!;
      const feat = this.#extractSparseFeatures(s.input);
      const pred = this.#forward(feat);
      if (pred === s.targetClass) correct++;
      totalLoss += -Math.log(Math.max(this.#probs[s.targetClass]!, LOG_EPS));
    }
    return {
      accuracy: correct / samples.length,
      correct,
      total: samples.length,
      meanLoss: totalLoss / samples.length,
    };
  }

  /**
   * Fast few-step online update to immediately crystallize a newly observed pattern.
   */
  crystallize(
    input: FeatureInput,
    targetClass: number,
    steps: number = 12,
    learningRate: number = 0.25,
  ): CrystallizeResult {
    if (!Number.isInteger(targetClass) || targetClass < 0 || targetClass >= this.numClasses) {
      throw new RangeError(`targetClass ${targetClass} out of range [0, ${this.numClasses - 1}]`);
    }
    if (!Number.isInteger(steps) || steps < 1) {
      throw new RangeError(`steps must be an integer >= 1, got ${steps}`);
    }

    const feat = this.#extractSparseFeatures(input);
    const frozenFeat: SparseFeatures = {
      indices: feat.indices.slice(0, feat.count),
      values: feat.values.slice(0, feat.count),
      count: feat.count,
    };

    const beforeClass = this.#forward(frozenFeat);
    const beforeConfidence = this.#probs[beforeClass]!;

    let finalLoss = 0;
    let stepsRun = 0;
    for (let s = 0; s < steps; s++) {
      finalLoss = this.#trainStep(frozenFeat, targetClass, learningRate, 0);
      stepsRun++;
    }

    const afterClass = this.#forward(frozenFeat);
    const afterConfidence = this.#probs[afterClass]!;

    return {
      beforeClass,
      afterClass,
      beforeConfidence,
      afterConfidence,
      stepsRun,
      finalLoss,
    };
  }

  /**
   * Serialize model into compact INT8 `.b8p` binary buffer (`B8SW` v1 / `B8G_ABI_MAGIC_V3`).
   */
  toB8pBuffer(): Uint8Array {
    const encoder = new TextEncoder();
    const labelBytes = encoder.encode(this.classLabels.join('\n'));

    const weightBytes =
      this.encRe.length +
      this.encIm.length +
      this.expRe.length +
      this.expIm.length +
      this.headWRe.length +
      this.headWIm.length +
      this.headBias.length;

    const totalBytes = B8SW_HEADER_BYTES + weightBytes + labelBytes.byteLength + 4;
    const buffer = new Uint8Array(totalBytes);
    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    const i8 = new Int8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);

    let offset = B8SW_HEADER_BYTES;
    const scaleEncRe = quantizeInt8Symmetric(this.encRe, i8, offset);
    offset += this.encRe.length;

    const scaleEncIm = quantizeInt8Symmetric(this.encIm, i8, offset);
    offset += this.encIm.length;

    const scaleExpRe = quantizeInt8Symmetric(this.expRe, i8, offset);
    offset += this.expRe.length;

    const scaleExpIm = quantizeInt8Symmetric(this.expIm, i8, offset);
    offset += this.expIm.length;

    const scaleHeadWRe = quantizeInt8Symmetric(this.headWRe, i8, offset);
    offset += this.headWRe.length;

    const scaleHeadWIm = quantizeInt8Symmetric(this.headWIm, i8, offset);
    offset += this.headWIm.length;

    const scaleHeadBias = quantizeInt8Symmetric(this.headBias, i8, offset);
    offset += this.headBias.length;

    buffer.set(labelBytes, offset);
    offset += labelBytes.byteLength;

    // Write 64-byte header
    buffer[0] = B8SW_MAGIC.charCodeAt(0);
    buffer[1] = B8SW_MAGIC.charCodeAt(1);
    buffer[2] = B8SW_MAGIC.charCodeAt(2);
    buffer[3] = B8SW_MAGIC.charCodeAt(3);
    view.setUint32(4, B8SW_VERSION, true);
    view.setUint32(8, B8G_ABI_MAGIC_V3, true);
    view.setUint32(12, this.vocabSize, true);
    view.setUint32(16, this.dPhasor, true);
    view.setUint32(20, this.numExperts, true);
    view.setUint32(24, this.topK, true);
    view.setUint32(28, this.numClasses, true);
    view.setFloat32(32, scaleEncRe, true);
    view.setFloat32(36, scaleEncIm, true);
    view.setFloat32(40, scaleExpRe, true);
    view.setFloat32(44, scaleExpIm, true);
    view.setFloat32(48, scaleHeadWRe, true);
    view.setFloat32(52, scaleHeadWIm, true);
    view.setFloat32(56, scaleHeadBias, true);
    view.setUint32(60, labelBytes.byteLength, true);

    // Trailer FNV-1a 32-bit checksum
    const checksum = fnv1a32Bytes(buffer, offset);
    view.setUint32(offset, checksum, true);

    return buffer;
  }

  /**
   * Deserialize an INT8-quantized `.b8p` binary buffer back into a runnable `SpinWaveMoEModel`.
   */
  static fromB8pBuffer(buffer: Uint8Array): SpinWaveMoEModel {
    const hdr = parseB8pHeader(buffer);
    const encLen = hdr.vocabSize * hdr.dPhasor;
    const expLen = hdr.numExperts * hdr.dPhasor;
    const headWLen = hdr.numExperts * hdr.numClasses * hdr.dPhasor;
    const headBLen = hdr.numExperts * hdr.numClasses;
    const expectedBytes =
      B8SW_HEADER_BYTES +
      encLen * 2 +
      expLen * 2 +
      headWLen * 2 +
      headBLen +
      hdr.labelsByteLength +
      4;

    if (buffer.byteLength !== expectedBytes) {
      throw new RangeError(
        `Corrupt B8SW .b8p buffer length: expected ${expectedBytes} bytes, got ${buffer.byteLength}`,
      );
    }

    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    const payloadEnd = buffer.byteLength - 4;
    const storedChecksum = view.getUint32(payloadEnd, true);
    const computedChecksum = fnv1a32Bytes(buffer, payloadEnd);
    if (storedChecksum !== computedChecksum) {
      throw new Error(
        `B8SW .b8p checksum mismatch: expected 0x${storedChecksum.toString(16)}, computed 0x${computedChecksum.toString(16)}`,
      );
    }

    const labelsStart = payloadEnd - hdr.labelsByteLength;
    const labelsText = new TextDecoder().decode(buffer.subarray(labelsStart, payloadEnd));
    const classLabels =
      hdr.labelsByteLength > 0
        ? labelsText.split('\n')
        : Array.from({ length: hdr.numClasses }, (_, i) => `class-${i}`);

    const model = new SpinWaveMoEModel({
      vocabSize: hdr.vocabSize,
      dPhasor: hdr.dPhasor,
      numExperts: hdr.numExperts,
      topK: hdr.topK,
      numClasses: hdr.numClasses,
      classLabels,
    });

    const i8 = new Int8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    let offset = B8SW_HEADER_BYTES;

    dequantizeInt8Symmetric(i8, offset, encLen, hdr.scales.encRe, model.encRe);
    offset += encLen;

    dequantizeInt8Symmetric(i8, offset, encLen, hdr.scales.encIm, model.encIm);
    offset += encLen;

    dequantizeInt8Symmetric(i8, offset, expLen, hdr.scales.expRe, model.expRe);
    offset += expLen;

    dequantizeInt8Symmetric(i8, offset, expLen, hdr.scales.expIm, model.expIm);
    offset += expLen;

    // Re-project dequantized expert keys strictly onto the unit circle (re^2 + im^2 = 1)
    for (let i = 0; i < expLen; i++) {
      const r = model.expRe[i]!;
      const im = model.expIm[i]!;
      const inv = 1 / Math.sqrt(r * r + im * im + NORM_EPS);
      model.expRe[i] = r * inv;
      model.expIm[i] = im * inv;
    }

    dequantizeInt8Symmetric(i8, offset, headWLen, hdr.scales.headWRe, model.headWRe);
    offset += headWLen;

    dequantizeInt8Symmetric(i8, offset, headWLen, hdr.scales.headWIm, model.headWIm);
    offset += headWLen;

    dequantizeInt8Symmetric(i8, offset, headBLen, hdr.scales.headBias, model.headBias);

    return model;
  }
}
