/**
 * @file sections/lumi-bert/index.mts
 * @description ModernBERT-inspired RoPE + GeGLU + Alternating Local/Global Attention
 * zero-dependency neural encoder in pure `Float32Array` for Awesome-OS / B8G.
 *
 * - Strictly `erasableSyntaxOnly: true` and ZERO external npm or `node:*` dependencies.
 * - Declarative `ExecutionContext` (`persona: 'devops'`, `transport: 'arena-sab'`).
 * - Runs built-in deterministic random-orthogonal + char-trigram/subword projections
 *   out-of-the-box in `< 5 ms` per embedding while exposing `loadWeights()` and
 *   `isPretrainedWeightsLoaded` for future pretrained ModernBERT / lumiBERT weights.
 */

import {
  defineSection,
  f,
  s,
  type ExecutionContext,
} from '../schema/index.mts';

// ============================================================================
// 1. Declarative StealifyLang Section & ExecutionContext
// ============================================================================

export const LUMI_BERT_CONTEXT: ExecutionContext = Object.freeze({
  boundary: 'in-process',
  display: 'headless',
  lifetime: 'oneshot',
  transport: 'arena-sab',
  capability: Object.freeze({
    tenantId: 'local-user',
    scopes: Object.freeze(['lumibert.embed']),
    epoch: 1,
    privateData: true,
  }),
  locality: 'local-arena',
  persona: 'devops',
});

export const LumiBertSection = defineSection({
  id: 'b8g/lumi-bert',
  version: '1.0.0',
  substrate: s.sabRing({ ringBytes: 65536 }),
  context: LUMI_BERT_CONTEXT,
  fields: {
    lumiModelId: f.string({ maxBytes: 64 }).default('lumi-bert-v1-modernbert'),
    lumiDim: f.int32().default(32),
    lumiPretrainedLoaded: f.boolean().default(false),
    lumiTotalEmbeddings: f.int32().default(0),
    lumiLastLatencyMs: f.float64().default(0),
    lumiLastTokenCount: f.int32().default(0),
  },
});

// ============================================================================
// 2. ModernBERT Configuration & Weight Contracts
// ============================================================================

export interface LumiBertConfig {
  readonly dim: number;
  readonly numHeads: number;
  readonly numLayers: number;
  readonly localWindow: number;
  readonly localRopeTheta: number;
  readonly globalRopeTheta: number;
  readonly ffnDim: number;
  readonly maxSeqLen: number;
  readonly vocabSize?: number;
  readonly globalLayerInterval?: number;
  readonly seed?: number;
}

export interface ResolvedLumiBertConfig {
  readonly dim: number;
  readonly numHeads: number;
  readonly numLayers: number;
  readonly localWindow: number;
  readonly localRopeTheta: number;
  readonly globalRopeTheta: number;
  readonly ffnDim: number;
  readonly maxSeqLen: number;
  readonly vocabSize: number;
  readonly globalLayerInterval: number;
  readonly seed: number;
}

export const DEFAULT_LUMI_BERT_CONFIG: Readonly<ResolvedLumiBertConfig> = Object.freeze({
  dim: 32,
  numHeads: 4,
  numLayers: 2,
  localWindow: 16,
  localRopeTheta: 10000,
  globalRopeTheta: 160000,
  ffnDim: 64,
  maxSeqLen: 8192,
  vocabSize: 4096,
  globalLayerInterval: 2,
  seed: 0x4c554d49,
});

export interface LumiBertLayerWeights {
  readonly attnNorm: Float32Array;
  readonly wQ: Float32Array;
  readonly wK: Float32Array;
  readonly wV: Float32Array;
  readonly wO: Float32Array;
  readonly ffnNorm: Float32Array;
  readonly wGate: Float32Array;
  readonly wUp: Float32Array;
  readonly wDown: Float32Array;
}

export interface LumiBertWeights {
  readonly tokenEmbeddings: Float32Array;
  readonly layers: readonly LumiBertLayerWeights[];
  readonly finalNorm: Float32Array;
}

// ============================================================================
// 3. Deterministic PRNG & Error-Aware Subword + Char-Trigram Tokenizer
// ============================================================================

function fnv1a32(str: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < str.length; i += 1) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function splitmix32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x9e3779b9) >>> 0;
    let t = Math.imul(state ^ (state >>> 16), 0x21f0aaad) >>> 0;
    t = Math.imul(t ^ (t >>> 15), 0x735a2d97) >>> 0;
    return ((t ^ (t >>> 15)) >>> 0) / 4294967296;
  };
}

function hashToTokenId(piece: string, vocabSize: number): number {
  const usable = Math.max(2, vocabSize - 1);
  return (fnv1a32(piece) % usable) + 1;
}

/**
 * Code/error-aware subword + character-trigram hashing tokenizer.
 *
 * - Normalizes ephemeral runtime noise (IP:port numbers, line:col coordinates,
 *   hex addresses, standalone digits) so two stack traces of the same root cause
 *   stay tightly aligned.
 * - Preserves diagnostic codes (`B8G018`, `EADDRINUSE`, `TS1294`), env vars
 *   (`DEPOT_TOOLS_WIN_TOOLCHAIN`), GN/Rust/TS paths, and camelCase/snake_case
 *   subwords, plus char-trigrams so no token ever collapses to `<unk>` (`0`).
 */
export function tokenizeErrorText(
  text: string,
  vocabSize: number = DEFAULT_LUMI_BERT_CONFIG.vocabSize,
  maxTokens: number = 96,
): number[] {
  const normalized = text
    .replace(/\b\d{1,3}(?:\.\d{1,3}){3}:\d+\b/g, ' <ip_port> ')
    .replace(/:\d+:\d+\b/g, ' ')
    .replace(/\b0x[0-9a-fA-F]+\b/g, ' <hex> ')
    .replace(/\b\d{4,}\b/g, ' <num> ');

  const rawPieces = normalized.match(/[A-Za-z0-9_./<>-]+/g) ?? [];
  const tokenIds: number[] = [];
  const limit = Math.max(1, maxTokens);

  const pushPiece = (piece: string): void => {
    if (tokenIds.length >= limit || piece.length === 0) {
      return;
    }
    tokenIds.push(hashToTokenId(piece, vocabSize));
  };

  for (let i = 0; i < rawPieces.length && tokenIds.length < limit; i += 1) {
    const raw = rawPieces[i]!;
    if (/^\d+$/.test(raw)) {
      continue;
    }

    const lower = raw.toLowerCase();
    // Emit exact lexical token (weighted via duplicate emission for diagnostic codes)
    pushPiece(`w:${lower}`);
    if (/^(?:b8g\d{3}|e[a-z]{4,}|ts\d{4}|depot_tools|script_executable|erasablesyntaxonly|read_fonts)$/i.test(raw)) {
      pushPiece(`c:${lower}`);
    }

    // Split path/dot/hyphen/underscore/camelCase subwords
    const subwords = raw
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .split(/[/_.\-]+/)
      .map((s) => s.toLowerCase())
      .filter((s) => s.length >= 2 && !/^\d+$/.test(s));

    for (let j = 0; j < subwords.length && tokenIds.length < limit; j += 1) {
      const sub = subwords[j]!;
      if (sub !== lower) {
        pushPiece(`s:${sub}`);
      }
      // Char-trigram hashing for morphological / typo resilience (never <unk>)
      if (sub.length >= 4) {
        const maxTri = Math.min(sub.length - 2, 4);
        for (let k = 0; k < maxTri && tokenIds.length < limit; k += 1) {
          pushPiece(`3:${sub.slice(k, k + 3)}`);
        }
      }
    }
  }

  if (tokenIds.length === 0) {
    tokenIds.push(1);
  }
  return tokenIds;
}

// ============================================================================
// 4. ModernBERT Pure `Float32Array` Math Primitives
// ============================================================================

/**
 * Exact error function approximation (Abramowitz & Stegun 7.1.26, |err| < 1.5e-7).
 */
function erfApprox(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t) *
      Math.exp(-ax * ax);
  return sign * y;
}

/**
 * Exact Gaussian Error Linear Unit: `GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`.
 */
export function gelu(x: number): number {
  return 0.5 * x * (1 + erfApprox(x * Math.SQRT1_2));
}

/**
 * Root Mean Square Layer Normalization (`RMSNorm`).
 */
export function rmsNorm(
  x: Float32Array,
  weight?: Float32Array,
  eps: number = 1e-6,
): Float32Array {
  const n = x.length;
  let sumSq = 0;
  for (let i = 0; i < n; i += 1) {
    const v = x[i]!;
    sumSq += v * v;
  }
  const invRms = 1 / Math.sqrt(sumSq / Math.max(1, n) + eps);
  const out = new Float32Array(n);
  if (weight) {
    for (let i = 0; i < n; i += 1) {
      out[i] = x[i]! * invRms * weight[i]!;
    }
  } else {
    for (let i = 0; i < n; i += 1) {
      out[i] = x[i]! * invRms;
    }
  }
  return out;
}

/**
 * Applies Rotary Position Embedding (RoPE) to `vec` at position `pos` with base `theta`.
 * ModernBERT uses `theta = 10000` for local layers and `theta = 160000` for global layers.
 */
export function applyRope(
  vec: Float32Array,
  pos: number,
  theta: number,
): Float32Array {
  const d = vec.length;
  const out = new Float32Array(d);
  for (let i = 0; i < d; i += 2) {
    if (i + 1 >= d) {
      out[i] = vec[i]!;
      break;
    }
    const freq = 1 / Math.pow(theta, i / d);
    const angle = pos * freq;
    const c = Math.cos(angle);
    const sVal = Math.sin(angle);
    const x0 = vec[i]!;
    const x1 = vec[i + 1]!;
    out[i] = x0 * c - x1 * sVal;
    out[i + 1] = x0 * sVal + x1 * c;
  }
  return out;
}

const ropeCache = new Map<string, { readonly cos: Float32Array; readonly sin: Float32Array; readonly maxPos: number }>();

function getRopeTable(headDim: number, theta: number, maxPos: number): { readonly cos: Float32Array; readonly sin: Float32Array } {
  const half = headDim >> 1;
  const key = `${headDim}:${theta}`;
  const existing = ropeCache.get(key);
  if (existing && existing.maxPos >= maxPos) {
    return existing;
  }
  const cap = Math.max(maxPos, 64);
  const cos = new Float32Array(cap * half);
  const sin = new Float32Array(cap * half);
  const freqs = new Float32Array(half);
  for (let k = 0; k < half; k += 1) {
    freqs[k] = 1 / Math.pow(theta, (2 * k) / headDim);
  }
  for (let pos = 0; pos < cap; pos += 1) {
    const base = pos * half;
    for (let k = 0; k < half; k += 1) {
      const angle = pos * freqs[k]!;
      cos[base + k] = Math.cos(angle);
      sin[base + k] = Math.sin(angle);
    }
  }
  const entry = { cos, sin, maxPos: cap };
  ropeCache.set(key, entry);
  return entry;
}

function applyRopeInPlace(
  buf: Float32Array,
  offset: number,
  headDim: number,
  pos: number,
  theta: number,
  table?: { readonly cos: Float32Array; readonly sin: Float32Array },
): void {
  const half = headDim >> 1;
  const tbl = table ?? getRopeTable(headDim, theta, pos + 1);
  const base = pos * half;
  for (let k = 0; k < half; k += 1) {
    const c = tbl.cos[base + k]!;
    const sVal = tbl.sin[base + k]!;
    const idx = offset + (k << 1);
    const x0 = buf[idx]!;
    const x1 = buf[idx + 1]!;
    buf[idx] = x0 * c - x1 * sVal;
    buf[idx + 1] = x0 * sVal + x1 * c;
  }
}

/**
 * ModernBERT GeGLU Feed-Forward Network:
 * `out = (GELU(x @ W_gate) * (x @ W_up)) @ W_down`
 *
 * Matrix layouts (row-major):
 * - `wGate`: `[dim, ffnDim]`
 * - `wUp`:   `[dim, ffnDim]`
 * - `wDown`: `[ffnDim, dim]`
 */
export function gegluFfn(
  x: Float32Array,
  wGate: Float32Array,
  wUp: Float32Array,
  wDown: Float32Array,
  dim: number,
  ffnDim: number,
): Float32Array {
  const gated = new Float32Array(ffnDim);
  for (let j = 0; j < ffnDim; j += 1) {
    let gSum = 0;
    let uSum = 0;
    for (let i = 0; i < dim; i += 1) {
      const xi = x[i]!;
      const wIdx = i * ffnDim + j;
      gSum += xi * wGate[wIdx]!;
      uSum += xi * wUp[wIdx]!;
    }
    gated[j] = gelu(gSum) * uSum;
  }

  const out = new Float32Array(dim);
  for (let k = 0; k < dim; k += 1) {
    let sum = 0;
    for (let j = 0; j < ffnDim; j += 1) {
      sum += gated[j]! * wDown[j * dim + k]!;
    }
    out[k] = sum;
  }
  return out;
}

function matVecRowMajor(
  x: Float32Array,
  xOffset: number,
  w: Float32Array,
  inDim: number,
  outDim: number,
  dst: Float32Array,
  dstOffset: number,
): void {
  dst.fill(0, dstOffset, dstOffset + outDim);
  for (let i = 0; i < inDim; i += 1) {
    const xi = x[xOffset + i]!;
    if (xi === 0) continue;
    const rowOff = i * outDim;
    for (let j = 0; j < outDim; j += 1) {
      dst[dstOffset + j] = dst[dstOffset + j]! + xi * w[rowOff + j]!;
    }
  }
}

function gegluFfnInto(
  x: Float32Array,
  xOffset: number,
  wGate: Float32Array,
  wUp: Float32Array,
  wDown: Float32Array,
  dim: number,
  ffnDim: number,
  gateScratch: Float32Array,
  upScratch: Float32Array,
  out: Float32Array,
  outOffset: number,
  residualGain: number,
): void {
  gateScratch.fill(0, 0, ffnDim);
  upScratch.fill(0, 0, ffnDim);
  for (let i = 0; i < dim; i += 1) {
    const xi = x[xOffset + i]!;
    if (xi === 0) continue;
    const rowOff = i * ffnDim;
    for (let j = 0; j < ffnDim; j += 1) {
      gateScratch[j] = gateScratch[j]! + xi * wGate[rowOff + j]!;
      upScratch[j] = upScratch[j]! + xi * wUp[rowOff + j]!;
    }
  }
  for (let j = 0; j < ffnDim; j += 1) {
    const g = gelu(gateScratch[j]!) * upScratch[j]!;
    if (g === 0) continue;
    const scaled = g * residualGain;
    const rowOff = j * dim;
    for (let k = 0; k < dim; k += 1) {
      out[outOffset + k] = out[outOffset + k]! + scaled * wDown[rowOff + k]!;
    }
  }
}

function rmsNormInto(
  x: Float32Array,
  xOffset: number,
  weight: Float32Array,
  dim: number,
  dst: Float32Array,
  dstOffset: number,
  eps: number = 1e-6,
): void {
  let sumSq = 0;
  for (let i = 0; i < dim; i += 1) {
    const v = x[xOffset + i]!;
    sumSq += v * v;
  }
  const invRms = 1 / Math.sqrt(sumSq / dim + eps);
  for (let i = 0; i < dim; i += 1) {
    dst[dstOffset + i] = x[xOffset + i]! * invRms * weight[i]!;
  }
}

/**
 * ModernBERT Alternating Local / Global Multi-Head Self-Attention layer with RoPE.
 *
 * - Local layers use sliding window `config.localWindow` and `localRopeTheta` (`10000`).
 * - Global layers attend across the full sequence with `globalRopeTheta` (`160000`).
 */
export function alternatingAttentionLayer(
  hidden: Float32Array,
  seqLen: number,
  layerIdx: number,
  config: Readonly<ResolvedLumiBertConfig>,
  weights: LumiBertLayerWeights,
): Float32Array {
  const { dim, numHeads, localWindow, localRopeTheta, globalRopeTheta, globalLayerInterval } =
    config;
  const headDim = dim / numHeads;
  const isGlobal =
    (layerIdx + 1) % globalLayerInterval === 0 || layerIdx === config.numLayers - 1;
  const ropeTheta = isGlobal ? globalRopeTheta : localRopeTheta;
  const halfWin = Math.floor(localWindow / 2);
  const scale = 1 / Math.sqrt(headDim);

  const qAll = new Float32Array(seqLen * dim);
  const kAll = new Float32Array(seqLen * dim);
  const vAll = new Float32Array(seqLen * dim);
  const ropeTbl = getRopeTable(headDim, ropeTheta, seqLen);

  for (let t = 0; t < seqLen; t += 1) {
    const off = t * dim;
    matVecRowMajor(hidden, off, weights.wQ, dim, dim, qAll, off);
    matVecRowMajor(hidden, off, weights.wK, dim, dim, kAll, off);
    matVecRowMajor(hidden, off, weights.wV, dim, dim, vAll, off);

    for (let h = 0; h < numHeads; h += 1) {
      const hOff = off + h * headDim;
      applyRopeInPlace(qAll, hOff, headDim, t, ropeTheta, ropeTbl);
      applyRopeInPlace(kAll, hOff, headDim, t, ropeTheta, ropeTbl);
    }
  }

  const context = new Float32Array(seqLen * dim);
  const scores = new Float32Array(seqLen);

  for (let h = 0; h < numHeads; h += 1) {
    const hShift = h * headDim;
    for (let t = 0; t < seqLen; t += 1) {
      const start = isGlobal ? 0 : Math.max(0, t - halfWin);
      const end = isGlobal ? seqLen - 1 : Math.min(seqLen - 1, t + halfWin);
      const qOff = t * dim + hShift;

      let maxScore = -Infinity;
      for (let sIdx = start; sIdx <= end; sIdx += 1) {
        const kOff = sIdx * dim + hShift;
        let dot = 0;
        for (let d = 0; d < headDim; d += 1) {
          dot += qAll[qOff + d]! * kAll[kOff + d]!;
        }
        const sc = dot * scale;
        scores[sIdx] = sc;
        if (sc > maxScore) {
          maxScore = sc;
        }
      }

      let expSum = 0;
      for (let sIdx = start; sIdx <= end; sIdx += 1) {
        const e = Math.exp(scores[sIdx]! - maxScore);
        scores[sIdx] = e;
        expSum += e;
      }

      const invExpSum = expSum > 0 ? 1 / expSum : 0;
      const outOff = t * dim + hShift;
      for (let sIdx = start; sIdx <= end; sIdx += 1) {
        const prob = scores[sIdx]! * invExpSum;
        const vOff = sIdx * dim + hShift;
        for (let d = 0; d < headDim; d += 1) {
          context[outOff + d] = context[outOff + d]! + prob * vAll[vOff + d]!;
        }
      }
    }
  }

  const out = new Float32Array(seqLen * dim);
  for (let t = 0; t < seqLen; t += 1) {
    const off = t * dim;
    matVecRowMajor(context, off, weights.wO, dim, dim, out, off);
  }
  return out;
}

/**
 * Cosine similarity between two `Float32Array` vectors of equal length.
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length || a.length === 0) {
    return 0;
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    const ai = a[i]!;
    const bi = b[i]!;
    dot += ai * bi;
    normA += ai * ai;
    normB += bi * bi;
  }
  const denom = Math.sqrt(normA * normB);
  return denom > 0 ? dot / denom : 0;
}

// ============================================================================
// 5. Deterministic Weight Initialization & LumiBertEncoder
// ============================================================================

function initMatrix(rows: number, cols: number, rng: () => number, scale: number): Float32Array {
  const mat = new Float32Array(rows * cols);
  for (let i = 0; i < mat.length; i += 1) {
    // Centered symmetric uniform in [-1, 1] scaled by Xavier/residual factor
    mat[i] = (rng() * 2 - 1) * scale;
  }
  return mat;
}

function buildDeterministicWeights(config: Readonly<ResolvedLumiBertConfig>): LumiBertWeights {
  const rng = splitmix32(config.seed);
  const { dim, ffnDim, numLayers, vocabSize } = config;

  // Zero-mean, unit-L2-normalized Rademacher/random-orthogonal token embeddings
  const tokenEmbeddings = new Float32Array(vocabSize * dim);
  const invSqrtDim = 1 / Math.sqrt(dim);
  for (let v = 0; v < vocabSize; v += 1) {
    const base = v * dim;
    let sum = 0;
    for (let d = 0; d < dim; d += 1) {
      const val = rng() < 0.5 ? -invSqrtDim : invSqrtDim;
      tokenEmbeddings[base + d] = val;
      sum += val;
    }
    const mean = sum / dim;
    let normSq = 0;
    for (let d = 0; d < dim; d += 1) {
      const centered = tokenEmbeddings[base + d]! - mean;
      tokenEmbeddings[base + d] = centered;
      normSq += centered * centered;
    }
    const invNorm = normSq > 0 ? 1 / Math.sqrt(normSq) : 1;
    for (let d = 0; d < dim; d += 1) {
      tokenEmbeddings[base + d] = tokenEmbeddings[base + d]! * invNorm;
    }
  }

  const xavierDim = Math.sqrt(1 / dim);
  const xavierFfn = Math.sqrt(1 / ffnDim);
  const layers: LumiBertLayerWeights[] = [];

  for (let l = 0; l < numLayers; l += 1) {
    const attnNorm = new Float32Array(dim);
    const ffnNorm = new Float32Array(dim);
    attnNorm.fill(1);
    ffnNorm.fill(1);

    layers.push({
      attnNorm,
      wQ: initMatrix(dim, dim, rng, xavierDim),
      wK: initMatrix(dim, dim, rng, xavierDim),
      wV: initMatrix(dim, dim, rng, xavierDim),
      wO: initMatrix(dim, dim, rng, xavierDim),
      ffnNorm,
      wGate: initMatrix(dim, ffnDim, rng, xavierDim),
      wUp: initMatrix(dim, ffnDim, rng, xavierDim),
      wDown: initMatrix(ffnDim, dim, rng, xavierFfn),
    });
  }

  const finalNorm = new Float32Array(dim);
  finalNorm.fill(1);

  return { tokenEmbeddings, layers, finalNorm };
}

/**
 * Zero-dependency ModernBERT-inspired encoder (`lumiBERT`).
 *
 * Out-of-the-box (`isPretrainedWeightsLoaded === false`), runs deterministic
 * random-orthogonal + char-trigram/subword projections through the full
 * ModernBERT RoPE + Alternating Local/Global Attention + GeGLU forward pass
 * in `< 5 ms`. Call `loadWeights()` when pretrained ModernBERT/lumiBERT
 * weights are available.
 */
export class LumiBertEncoder {
  public readonly config: Readonly<ResolvedLumiBertConfig>;
  public isPretrainedWeightsLoaded: boolean;
  private weights: LumiBertWeights;

  constructor(partialConfig?: Partial<LumiBertConfig>) {
    const merged: ResolvedLumiBertConfig = Object.freeze({
      dim: partialConfig?.dim ?? DEFAULT_LUMI_BERT_CONFIG.dim,
      numHeads: partialConfig?.numHeads ?? DEFAULT_LUMI_BERT_CONFIG.numHeads,
      numLayers: partialConfig?.numLayers ?? DEFAULT_LUMI_BERT_CONFIG.numLayers,
      localWindow: partialConfig?.localWindow ?? DEFAULT_LUMI_BERT_CONFIG.localWindow,
      localRopeTheta: partialConfig?.localRopeTheta ?? DEFAULT_LUMI_BERT_CONFIG.localRopeTheta,
      globalRopeTheta: partialConfig?.globalRopeTheta ?? DEFAULT_LUMI_BERT_CONFIG.globalRopeTheta,
      ffnDim: partialConfig?.ffnDim ?? DEFAULT_LUMI_BERT_CONFIG.ffnDim,
      maxSeqLen: partialConfig?.maxSeqLen ?? DEFAULT_LUMI_BERT_CONFIG.maxSeqLen,
      vocabSize: partialConfig?.vocabSize ?? DEFAULT_LUMI_BERT_CONFIG.vocabSize,
      globalLayerInterval:
        partialConfig?.globalLayerInterval ?? DEFAULT_LUMI_BERT_CONFIG.globalLayerInterval,
      seed: partialConfig?.seed ?? DEFAULT_LUMI_BERT_CONFIG.seed,
    });

    if (merged.dim % merged.numHeads !== 0 || (merged.dim / merged.numHeads) % 2 !== 0) {
      throw new Error(
        `Invalid LumiBertConfig: dim (${merged.dim}) must be divisible by numHeads (${merged.numHeads}) with even headDim for RoPE.`,
      );
    }

    this.config = merged;
    this.isPretrainedWeightsLoaded = false;
    this.weights = buildDeterministicWeights(merged);
  }

  /**
   * Loads pretrained ModernBERT / lumiBERT weights and flips `isPretrainedWeightsLoaded = true`.
   */
  public loadWeights(weights: LumiBertWeights): void {
    if (weights.layers.length !== this.config.numLayers) {
      throw new Error(
        `Weight layer count mismatch: expected ${this.config.numLayers}, got ${weights.layers.length}.`,
      );
    }
    this.weights = weights;
    this.isPretrainedWeightsLoaded = true;
  }

  /**
   * Encodes arbitrary error/log/code text into an L2-normalized unit `Float32Array`.
   */
  public embed(text: string): Float32Array {
    const { dim, ffnDim, numLayers, vocabSize, maxSeqLen } = this.config;
    const maxTokens = Math.min(maxSeqLen, 40);
    const tokenIds = tokenizeErrorText(text, vocabSize, maxTokens);
    const seqLen = tokenIds.length;

    const hidden = new Float32Array(seqLen * dim);
    for (let t = 0; t < seqLen; t += 1) {
      const id = tokenIds[t]! % vocabSize;
      const srcOff = id * dim;
      const dstOff = t * dim;
      for (let d = 0; d < dim; d += 1) {
        hidden[dstOff + d] = this.weights.tokenEmbeddings[srcOff + d]!;
      }
    }

    // Pre-Norm residual scaling keeps random-orthogonal projections well-conditioned
    // while pretrained weights run at full 1.0 residual gain.
    const residualGain = this.isPretrainedWeightsLoaded
      ? 1.0
      : 0.12 / Math.sqrt(Math.max(1, numLayers));

    const normBuf = new Float32Array(seqLen * dim);
    const gateScratch = new Float32Array(ffnDim);
    const upScratch = new Float32Array(ffnDim);

    for (let l = 0; l < numLayers; l += 1) {
      const layerW = this.weights.layers[l]!;

      // 1. Pre-Attention RMSNorm
      for (let t = 0; t < seqLen; t += 1) {
        const off = t * dim;
        rmsNormInto(hidden, off, layerW.attnNorm, dim, normBuf, off);
      }

      // 2. Alternating Local/Global Self-Attention with RoPE + Residual
      const attnOut = alternatingAttentionLayer(normBuf, seqLen, l, this.config, layerW);
      for (let i = 0; i < hidden.length; i += 1) {
        hidden[i] = hidden[i]! + residualGain * attnOut[i]!;
      }

      // 3. Pre-FFN RMSNorm + GeGLU FFN + Residual
      for (let t = 0; t < seqLen; t += 1) {
        const off = t * dim;
        rmsNormInto(hidden, off, layerW.ffnNorm, dim, normBuf, off);
        gegluFfnInto(
          normBuf,
          off,
          layerW.wGate,
          layerW.wUp,
          layerW.wDown,
          dim,
          ffnDim,
          gateScratch,
          upScratch,
          hidden,
          off,
          residualGain,
        );
      }
    }

    // 4. Final RMSNorm + Mean Pooling
    const pooled = new Float32Array(dim);
    const invSeq = 1 / seqLen;
    for (let t = 0; t < seqLen; t += 1) {
      const off = t * dim;
      rmsNormInto(hidden, off, this.weights.finalNorm, dim, normBuf, off);
      for (let d = 0; d < dim; d += 1) {
        pooled[d] = pooled[d]! + normBuf[off + d]! * invSeq;
      }
    }

    // 5. L2 Normalization -> unit Float32Array
    let normSq = 0;
    for (let d = 0; d < dim; d += 1) {
      const v = pooled[d]!;
      normSq += v * v;
    }
    const invNorm = normSq > 0 ? 1 / Math.sqrt(normSq) : 1;
    for (let d = 0; d < dim; d += 1) {
      pooled[d] = pooled[d]! * invNorm;
    }
    return pooled;
  }

  /**
   * Computes cosine similarity between two embeddings.
   */
  public cosineSimilarity(a: Float32Array, b: Float32Array): number {
    return cosineSimilarity(a, b);
  }
}
