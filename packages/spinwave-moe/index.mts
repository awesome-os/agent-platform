/**
 * Spin-Wave Computing Intersection Bus (`B8G_ARENA_V1` / `B8G_ABI_MAGIC_V3`)
 *
 * Zero-copy, lock-free magnonic phasor superposition bus backed by a single
 * 64-byte cache-line-aligned `SharedArrayBuffer`.
 *
 * Binary Layout (little-endian, 64-byte aligned):
 *   Header [bytes 0..63] (16 x 32-bit words):
 *     word 0  (offset  0): magic       = 0x33564742 (B8G_ABI_MAGIC_V3)
 *     word 1  (offset  4): version     = 1
 *     word 2  (offset  8): numBands    >= 1
 *     word 3  (offset 12): stride      = 64 (bytes per band cell)
 *     word 4  (offset 16): globalEpoch (Int32 atomic counter)
 *     word 5  (offset 20): flags       (reserved = 0)
 *     word 6  (offset 24): defaultGamma (Float32)
 *     words 7..15: reserved padding (64-byte alignment)
 *
 *   Band Cell `b` [bytes 64 + b * 64 .. 64 + (b + 1) * 64 - 1]:
 *     word +0 (f32): re            (real component of complex phasor)
 *     word +1 (f32): im            (imaginary component of complex phasor)
 *     word +2 (f32): dampingGamma  (Gilbert damping constant gamma >= 0)
 *     word +3 (f32): energy        (instantaneous energy |psi|^2 = re^2 + im^2)
 *     word +4 (f32): lastTick      (accumulated simulation time / tick)
 *     word +5 (f32): peakAmplitude (peak amplitude observed since last clear)
 *     word +6 (f32): frequencyTag  (orthogonal FDM carrier band identifier)
 *     word +7 (f32): reserved
 *     word +8 (i32): epoch         (atomic seqlock: even = consistent, odd = writing)
 *     word +9 (i32): emitCount     (atomic wave emission counter)
 *     words +10..+15: cache-line padding to 64 bytes (zero false sharing)
 */

export const B8G_ABI_MAGIC_V3 = 0x33564742;
export const SPINWAVE_ARENA_VERSION = 1;
export const HEADER_BYTES = 64;
export const CELL_STRIDE_BYTES = 64;
export const WORDS_PER_LINE = 16;

// Header 32-bit word offsets
const HDR_MAGIC = 0;
const HDR_VERSION = 1;
const HDR_NUM_BANDS = 2;
const HDR_STRIDE = 3;
const HDR_GLOBAL_EPOCH = 4;
const HDR_FLAGS = 5;
const HDR_DEFAULT_GAMMA = 6;

// Per-band cell 32-bit word offsets (relative to cell base word)
const CELL_RE = 0;
const CELL_IM = 1;
const CELL_DAMPING_GAMMA = 2;
const CELL_ENERGY = 3;
const CELL_LAST_TICK = 4;
const CELL_PEAK_AMPLITUDE = 5;
const CELL_FREQUENCY_TAG = 6;
const CELL_EPOCH = 8;
const CELL_EMIT_COUNT = 9;

export type MagnonicBit = -1 | 1;

export interface SpinWaveArenaOptions {
  readonly numBands?: number;
  readonly defaultDampingGamma?: number;
}

export interface SpinWaveSample {
  readonly band: number;
  readonly re: number;
  readonly im: number;
  readonly amplitude: number;
  readonly phaseRad: number;
  readonly coherentSign: MagnonicBit;
  readonly energy: number;
  readonly dampingGamma: number;
  readonly lastTick: number;
  readonly epoch: number;
}

export interface ExpertRouteResult {
  readonly index: number;
  readonly expertPhaseRad: number;
  readonly coherence: number;
}

export class SpinWaveArenaBus {
  readonly buffer: SharedArrayBuffer;
  readonly numBands: number;
  readonly stride: number;
  readonly magic: number;
  readonly version: number;

  readonly #i32: Int32Array;
  readonly #f32: Float32Array;

  constructor(optionsOrBuffer?: SpinWaveArenaOptions | SharedArrayBuffer) {
    if (optionsOrBuffer instanceof SharedArrayBuffer) {
      const sab = optionsOrBuffer;
      if (sab.byteLength < HEADER_BYTES) {
        throw new RangeError(
          `SharedArrayBuffer too small for B8G_ARENA_V1 header: ${sab.byteLength} < ${HEADER_BYTES}`,
        );
      }

      const i32 = new Int32Array(sab);
      const magic = Atomics.load(i32, HDR_MAGIC);
      if (magic !== B8G_ABI_MAGIC_V3) {
        throw new Error(
          `Invalid B8G_ABI_MAGIC_V3 header: expected 0x${B8G_ABI_MAGIC_V3.toString(16)}, got 0x${(magic >>> 0).toString(16)}`,
        );
      }

      const version = Atomics.load(i32, HDR_VERSION);
      if (version !== SPINWAVE_ARENA_VERSION) {
        throw new Error(
          `Unsupported SpinWaveArenaBus version: expected ${SPINWAVE_ARENA_VERSION}, got ${version}`,
        );
      }

      const numBands = Atomics.load(i32, HDR_NUM_BANDS);
      const stride = Atomics.load(i32, HDR_STRIDE);
      if (numBands < 1 || stride !== CELL_STRIDE_BYTES) {
        throw new Error(
          `Invalid SpinWaveArenaBus geometry: numBands=${numBands}, stride=${stride}`,
        );
      }

      const requiredBytes = HEADER_BYTES + numBands * stride;
      if (sab.byteLength < requiredBytes) {
        throw new RangeError(
          `SharedArrayBuffer truncated: ${sab.byteLength} < ${requiredBytes}`,
        );
      }

      this.buffer = sab;
      this.magic = magic;
      this.version = version;
      this.numBands = numBands;
      this.stride = stride;
      this.#i32 = i32;
      this.#f32 = new Float32Array(sab);
      return;
    }

    const numBands = optionsOrBuffer?.numBands ?? 8;
    const defaultDampingGamma = optionsOrBuffer?.defaultDampingGamma ?? 0.5;

    if (!Number.isInteger(numBands) || numBands < 1) {
      throw new RangeError(`numBands must be a positive integer, got ${numBands}`);
    }
    if (!Number.isFinite(defaultDampingGamma) || defaultDampingGamma < 0) {
      throw new RangeError(
        `defaultDampingGamma must be a finite non-negative number, got ${defaultDampingGamma}`,
      );
    }

    const totalBytes = HEADER_BYTES + numBands * CELL_STRIDE_BYTES;
    const sab = new SharedArrayBuffer(totalBytes);
    const i32 = new Int32Array(sab);
    const f32 = new Float32Array(sab);

    Atomics.store(i32, HDR_MAGIC, B8G_ABI_MAGIC_V3);
    Atomics.store(i32, HDR_VERSION, SPINWAVE_ARENA_VERSION);
    Atomics.store(i32, HDR_NUM_BANDS, numBands);
    Atomics.store(i32, HDR_STRIDE, CELL_STRIDE_BYTES);
    Atomics.store(i32, HDR_GLOBAL_EPOCH, 0);
    Atomics.store(i32, HDR_FLAGS, 0);
    f32[HDR_DEFAULT_GAMMA] = defaultDampingGamma;

    for (let b = 0; b < numBands; b++) {
      const base = WORDS_PER_LINE + b * WORDS_PER_LINE;
      f32[base + CELL_RE] = 0;
      f32[base + CELL_IM] = 0;
      f32[base + CELL_DAMPING_GAMMA] = defaultDampingGamma;
      f32[base + CELL_ENERGY] = 0;
      f32[base + CELL_LAST_TICK] = 0;
      f32[base + CELL_PEAK_AMPLITUDE] = 0;
      f32[base + CELL_FREQUENCY_TAG] = b + 1;
      Atomics.store(i32, base + CELL_EPOCH, 0);
      Atomics.store(i32, base + CELL_EMIT_COUNT, 0);
    }

    this.buffer = sab;
    this.magic = B8G_ABI_MAGIC_V3;
    this.version = SPINWAVE_ARENA_VERSION;
    this.numBands = numBands;
    this.stride = CELL_STRIDE_BYTES;
    this.#i32 = i32;
    this.#f32 = f32;
  }

  /**
   * Attaches a zero-copy view onto an existing 64-byte-aligned `SharedArrayBuffer`
   * previously initialized by another `SpinWaveArenaBus` instance or worker.
   */
  static attachExistingArena(sharedBuffer: SharedArrayBuffer): SpinWaveArenaBus {
    return new SpinWaveArenaBus(sharedBuffer);
  }

  /**
   * Instance convenience alias for attaching to an existing `SharedArrayBuffer`.
   */
  attachExistingArena(sharedBuffer: SharedArrayBuffer): SpinWaveArenaBus {
    return SpinWaveArenaBus.attachExistingArena(sharedBuffer);
  }

  #cellBaseWord(band: number): number {
    if (!Number.isInteger(band) || band < 0 || band >= this.numBands) {
      throw new RangeError(
        `Band index out of range: ${band} (valid: 0..${this.numBands - 1})`,
      );
    }
    return WORDS_PER_LINE + band * WORDS_PER_LINE;
  }

  /**
   * Acquires the per-band seqlock for writing (transitions epoch from even -> odd).
   */
  #beginWrite(base: number): number {
    const epochIdx = base + CELL_EPOCH;
    for (;;) {
      const epoch = Atomics.load(this.#i32, epochIdx);
      if ((epoch & 1) !== 0) {
        continue;
      }
      if (Atomics.compareExchange(this.#i32, epochIdx, epoch, epoch + 1) === epoch) {
        return epoch;
      }
    }
  }

  /**
   * Releases the per-band seqlock (transitions epoch from odd -> next even).
   */
  #endWrite(base: number, startEpoch: number): void {
    Atomics.store(this.#i32, base + CELL_EPOCH, startEpoch + 2);
  }

  /**
   * Injects a spin wave with complex phasor `A * exp(i * phaseRad)` into `band`:
   *   re += A * cos(phaseRad)
   *   im += A * sin(phaseRad)
   *
   * Orthogonal FDM bands occupy disjoint 64-byte cache lines and never collide.
   */
  emitWave(band: number, amplitude: number, phaseRad: number): void {
    if (!Number.isFinite(amplitude) || !Number.isFinite(phaseRad)) {
      throw new RangeError(
        `amplitude and phaseRad must be finite numbers, got (${amplitude}, ${phaseRad})`,
      );
    }

    const base = this.#cellBaseWord(band);
    const dRe = amplitude * Math.cos(phaseRad);
    const dIm = amplitude * Math.sin(phaseRad);

    const startEpoch = this.#beginWrite(base);
    const nextRe = this.#f32[base + CELL_RE] + dRe;
    const nextIm = this.#f32[base + CELL_IM] + dIm;
    const energy = nextRe * nextRe + nextIm * nextIm;
    const amp = Math.sqrt(energy);

    this.#f32[base + CELL_RE] = nextRe;
    this.#f32[base + CELL_IM] = nextIm;
    this.#f32[base + CELL_ENERGY] = energy;
    if (amp > this.#f32[base + CELL_PEAK_AMPLITUDE]) {
      this.#f32[base + CELL_PEAK_AMPLITUDE] = amp;
    }
    Atomics.add(this.#i32, base + CELL_EMIT_COUNT, 1);
    this.#endWrite(base, startEpoch);
  }

  /**
   * Lock-free seqlock snapshot of the superposed spin-wave intersection on `band`.
   */
  sampleIntersection(band: number): SpinWaveSample {
    const base = this.#cellBaseWord(band);
    const epochIdx = base + CELL_EPOCH;

    for (;;) {
      const e1 = Atomics.load(this.#i32, epochIdx);
      if ((e1 & 1) !== 0) {
        continue;
      }

      const re = this.#f32[base + CELL_RE];
      const im = this.#f32[base + CELL_IM];
      const dampingGamma = this.#f32[base + CELL_DAMPING_GAMMA];
      const energy = this.#f32[base + CELL_ENERGY];
      const lastTick = this.#f32[base + CELL_LAST_TICK];

      const e2 = Atomics.load(this.#i32, epochIdx);
      if (e1 === e2) {
        const amplitude = Math.hypot(re, im);
        const phaseRad = Math.atan2(im, re);
        const coherentSign: MagnonicBit = re >= 0 ? 1 : -1;
        return {
          band,
          re,
          im,
          amplitude,
          phaseRad,
          coherentSign,
          energy,
          dampingGamma,
          lastTick,
          epoch: e2,
        };
      }
    }
  }

  /**
   * True magnonic 3-input Majority Gate:
   * Encodes `a, b, c in {-1, +1}` as spin-wave phases `{pi, 0}`,
   * superposes them on `band`, and returns `re >= 0 ? 1 : -1`.
   */
  majorityGate(
    band: number,
    a: MagnonicBit,
    b: MagnonicBit,
    c: MagnonicBit,
  ): MagnonicBit {
    if ((a !== -1 && a !== 1) || (b !== -1 && b !== 1) || (c !== -1 && c !== 1)) {
      throw new RangeError(`majorityGate inputs must be -1 or +1, got (${a}, ${b}, ${c})`);
    }

    this.clearBand(band);
    this.emitWave(band, 1, a === 1 ? 0 : Math.PI);
    this.emitWave(band, 1, b === 1 ? 0 : Math.PI);
    this.emitWave(band, 1, c === 1 ? 0 : Math.PI);

    return this.sampleIntersection(band).coherentSign;
  }

  /**
   * Applies natural exponential Gilbert magnonic damping `exp(-gamma * dt)`
   * across all bands in-place (automatic backpressure, zero heap allocations).
   */
  stepDamping(dt: number): void {
    if (!Number.isFinite(dt) || dt < 0) {
      throw new RangeError(`dt must be a finite non-negative number, got ${dt}`);
    }
    if (dt === 0) {
      return;
    }

    for (let b = 0; b < this.numBands; b++) {
      const base = WORDS_PER_LINE + b * WORDS_PER_LINE;
      const startEpoch = this.#beginWrite(base);

      const gamma = this.#f32[base + CELL_DAMPING_GAMMA];
      const decay = Math.exp(-gamma * dt);
      const nextRe = this.#f32[base + CELL_RE] * decay;
      const nextIm = this.#f32[base + CELL_IM] * decay;

      this.#f32[base + CELL_RE] = nextRe;
      this.#f32[base + CELL_IM] = nextIm;
      this.#f32[base + CELL_ENERGY] = nextRe * nextRe + nextIm * nextIm;
      this.#f32[base + CELL_LAST_TICK] = this.#f32[base + CELL_LAST_TICK] + dt;

      this.#endWrite(base, startEpoch);
    }

    Atomics.add(this.#i32, HDR_GLOBAL_EPOCH, 1);
  }

  /**
   * Phase-coherence Mixture-of-Experts router using `cos(queryPhaseRad - expertPhase)`.
   * Returns the top-`k` experts sorted by descending coherence score.
   */
  routeTopKExperts(
    queryPhaseRad: number,
    expertPhases: ArrayLike<number>,
    k: number,
  ): ExpertRouteResult[] {
    if (!Number.isFinite(queryPhaseRad)) {
      throw new RangeError(`queryPhaseRad must be finite, got ${queryPhaseRad}`);
    }
    if (!Number.isInteger(k) || k < 0) {
      throw new RangeError(`k must be a non-negative integer, got ${k}`);
    }

    const count = expertPhases.length;
    const limit = Math.min(k, count);
    if (limit === 0) {
      return [];
    }

    const scored: ExpertRouteResult[] = new Array(count);
    for (let i = 0; i < count; i++) {
      const expertPhaseRad = expertPhases[i];
      const coherence = Math.cos(queryPhaseRad - expertPhaseRad);
      scored[i] = { index: i, expertPhaseRad, coherence };
    }

    scored.sort((lhs, rhs) => {
      if (rhs.coherence !== lhs.coherence) {
        return rhs.coherence - lhs.coherence;
      }
      return lhs.index - rhs.index;
    });

    return scored.slice(0, limit);
  }

  /**
   * Sets the Gilbert damping constant `gamma >= 0` for a specific band.
   */
  setBandDamping(band: number, gamma: number): void {
    if (!Number.isFinite(gamma) || gamma < 0) {
      throw new RangeError(`gamma must be a finite non-negative number, got ${gamma}`);
    }
    const base = this.#cellBaseWord(band);
    const startEpoch = this.#beginWrite(base);
    this.#f32[base + CELL_DAMPING_GAMMA] = gamma;
    this.#endWrite(base, startEpoch);
  }

  /**
   * Reads the current Gilbert damping constant `gamma` for a specific band.
   */
  getBandDamping(band: number): number {
    return this.sampleIntersection(band).dampingGamma;
  }

  /**
   * Returns the atomic seqlock epoch counter for `band`.
   */
  getBandEpoch(band: number): number {
    const base = this.#cellBaseWord(band);
    return Atomics.load(this.#i32, base + CELL_EPOCH);
  }

  /**
   * Returns the global arena epoch counter.
   */
  getGlobalEpoch(): number {
    return Atomics.load(this.#i32, HDR_GLOBAL_EPOCH);
  }

  /**
   * Zeros the complex phasor and energy of a single band while preserving its
   * damping constant and advancing its atomic epoch.
   */
  clearBand(band: number): void {
    const base = this.#cellBaseWord(band);
    const startEpoch = this.#beginWrite(base);
    this.#f32[base + CELL_RE] = 0;
    this.#f32[base + CELL_IM] = 0;
    this.#f32[base + CELL_ENERGY] = 0;
    this.#f32[base + CELL_PEAK_AMPLITUDE] = 0;
    this.#endWrite(base, startEpoch);
  }

  /**
   * Clears all bands in the arena and increments the global epoch.
   */
  resetAll(): void {
    for (let b = 0; b < this.numBands; b++) {
      this.clearBand(b);
    }
    Atomics.add(this.#i32, HDR_GLOBAL_EPOCH, 1);
  }
}

/**
 * Convenience top-level wrapper for zero-copy attachment to an existing arena.
 */
export function attachExistingArena(sharedBuffer: SharedArrayBuffer): SpinWaveArenaBus {
  return SpinWaveArenaBus.attachExistingArena(sharedBuffer);
}
