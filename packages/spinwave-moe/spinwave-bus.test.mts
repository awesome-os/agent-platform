import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  B8G_ABI_MAGIC_V3,
  CELL_STRIDE_BYTES,
  HEADER_BYTES,
  SPINWAVE_ARENA_VERSION,
  SpinWaveArenaBus,
  attachExistingArena,
  type MagnonicBit,
} from './index.mts';

describe('SpinWaveArenaBus (B8G_ARENA_V1 / B8G_ABI_MAGIC_V3)', () => {
  it('initializes a 64-byte-aligned SharedArrayBuffer header and band cells', () => {
    const bus = new SpinWaveArenaBus({ numBands: 8, defaultDampingGamma: 0.25 });

    assert.ok(bus.buffer instanceof SharedArrayBuffer);
    assert.equal(bus.buffer.byteLength, HEADER_BYTES + 8 * CELL_STRIDE_BYTES);
    assert.equal(bus.buffer.byteLength % 64, 0);

    const header = new Int32Array(bus.buffer, 0, 16);
    assert.equal(Atomics.load(header, 0), B8G_ABI_MAGIC_V3);
    assert.equal(Atomics.load(header, 1), SPINWAVE_ARENA_VERSION);
    assert.equal(Atomics.load(header, 2), 8);
    assert.equal(Atomics.load(header, 3), 64);

    const sample0 = bus.sampleIntersection(0);
    assert.equal(sample0.re, 0);
    assert.equal(sample0.im, 0);
    assert.equal(sample0.amplitude, 0);
    assert.equal(sample0.dampingGamma, 0.25);
    assert.equal(sample0.epoch, 0);
  });

  it('verifies ALL 8 truth-table rows of the magnonic 3-input Majority Gate', () => {
    const bus = new SpinWaveArenaBus({ numBands: 4 });
    const truthTable: ReadonlyArray<[MagnonicBit, MagnonicBit, MagnonicBit, MagnonicBit]> = [
      [-1, -1, -1, -1],
      [-1, -1, 1, -1],
      [-1, 1, -1, -1],
      [-1, 1, 1, 1],
      [1, -1, -1, -1],
      [1, -1, 1, 1],
      [1, 1, -1, 1],
      [1, 1, 1, 1],
    ];

    for (const [a, b, c, expected] of truthTable) {
      const actual = bus.majorityGate(0, a, b, c);
      assert.equal(
        actual,
        expected,
        `majorityGate(${a}, ${b}, ${c}) expected ${expected}, got ${actual}`,
      );

      const sample = bus.sampleIntersection(0);
      assert.equal(sample.coherentSign, expected);
      assert.ok(
        Math.abs(sample.re - (a + b + c)) < 1e-5,
        `expected superposed re=${a + b + c}, got ${sample.re}`,
      );
    }
  });

  it('demonstrates constructive (phi=0 + phi=0 -> 2A) and destructive (phi=0 + phi=pi -> 0) interference', () => {
    const bus = new SpinWaveArenaBus({ numBands: 2 });
    const amplitude = 3.5;

    // Constructive interference on band 0
    bus.emitWave(0, amplitude, 0);
    bus.emitWave(0, amplitude, 0);
    const constructive = bus.sampleIntersection(0);
    assert.ok(
      Math.abs(constructive.re - 2 * amplitude) < 1e-5,
      `expected re=${2 * amplitude}, got ${constructive.re}`,
    );
    assert.ok(Math.abs(constructive.im) < 1e-5);
    assert.ok(Math.abs(constructive.amplitude - 2 * amplitude) < 1e-5);
    assert.equal(constructive.coherentSign, 1);

    // Destructive interference on band 1
    bus.emitWave(1, amplitude, 0);
    bus.emitWave(1, amplitude, Math.PI);
    const destructive = bus.sampleIntersection(1);
    assert.ok(
      Math.abs(destructive.re) < 1e-5,
      `expected re~0, got ${destructive.re}`,
    );
    assert.ok(
      Math.abs(destructive.im) < 1e-5,
      `expected im~0, got ${destructive.im}`,
    );
    assert.ok(
      destructive.amplitude < 1e-5,
      `expected amplitude~0, got ${destructive.amplitude}`,
    );
  });

  it('supports Frequency-Division Multiplexing across 8 orthogonal bands with zero crosstalk', () => {
    const numBands = 8;
    const bus = new SpinWaveArenaBus({ numBands });

    const expectedAmps = [1.0, 2.5, 4.0, 0.75, 3.25, 5.5, 1.5, 6.0];
    const expectedPhases = [
      0,
      Math.PI / 6,
      Math.PI / 3,
      Math.PI / 2,
      (2 * Math.PI) / 3,
      (5 * Math.PI) / 6,
      Math.PI,
      -Math.PI / 4,
    ];

    for (let band = 0; band < numBands; band++) {
      bus.emitWave(band, expectedAmps[band], expectedPhases[band]);
    }

    // Mutate band 3 heavily and ensure no other band is disturbed
    bus.emitWave(3, 10.0, 0);

    for (let band = 0; band < numBands; band++) {
      const sample = bus.sampleIntersection(band);
      if (band === 3) {
        const expRe = expectedAmps[3] * Math.cos(expectedPhases[3]) + 10.0;
        const expIm = expectedAmps[3] * Math.sin(expectedPhases[3]);
        assert.ok(Math.abs(sample.re - expRe) < 1e-4);
        assert.ok(Math.abs(sample.im - expIm) < 1e-4);
        assert.equal(sample.epoch, 4); // 2 emits * 2 seqlock increments
      } else {
        const expRe = expectedAmps[band] * Math.cos(expectedPhases[band]);
        const expIm = expectedAmps[band] * Math.sin(expectedPhases[band]);
        assert.ok(
          Math.abs(sample.re - expRe) < 1e-4,
          `band ${band} crosstalk on re: expected ${expRe}, got ${sample.re}`,
        );
        assert.ok(
          Math.abs(sample.im - expIm) < 1e-4,
          `band ${band} crosstalk on im: expected ${expIm}, got ${sample.im}`,
        );
        assert.ok(Math.abs(sample.amplitude - expectedAmps[band]) < 1e-4);
        assert.equal(sample.epoch, 2); // 1 emit * 2 seqlock increments
      }
    }
  });

  it('applies damped-gravity exponential decay exp(-gamma * dt) to equilibrium', () => {
    const bus = new SpinWaveArenaBus({ numBands: 2, defaultDampingGamma: 1.0 });
    bus.setBandDamping(1, 0.25);

    bus.emitWave(0, 10.0, Math.PI / 4);
    bus.emitWave(1, 10.0, Math.PI / 4);

    // Single damping step dt = 1.0
    bus.stepDamping(1.0);
    const s0After1 = bus.sampleIntersection(0);
    const s1After1 = bus.sampleIntersection(1);

    assert.ok(
      Math.abs(s0After1.amplitude - 10.0 * Math.exp(-1.0)) < 1e-4,
      `band 0 expected ${10.0 * Math.exp(-1.0)}, got ${s0After1.amplitude}`,
    );
    assert.ok(
      Math.abs(s1After1.amplitude - 10.0 * Math.exp(-0.25)) < 1e-4,
      `band 1 expected ${10.0 * Math.exp(-0.25)}, got ${s1After1.amplitude}`,
    );
    assert.ok(s0After1.amplitude < s1After1.amplitude);
    assert.equal(s0After1.lastTick, 1.0);
    assert.equal(bus.getGlobalEpoch(), 1);

    // Decay to equilibrium over 40 time units
    for (let step = 0; step < 40; step++) {
      bus.stepDamping(1.0);
    }
    const equilibrium0 = bus.sampleIntersection(0);
    const equilibrium1 = bus.sampleIntersection(1);

    assert.ok(
      equilibrium0.amplitude < 1e-12,
      `expected band 0 equilibrium < 1e-12, got ${equilibrium0.amplitude}`,
    );
    assert.ok(
      equilibrium1.amplitude < 1e-3,
      `expected band 1 equilibrium < 1e-3, got ${equilibrium1.amplitude}`,
    );
    assert.equal(equilibrium0.lastTick, 41.0);
  });

  it('routes Top-K MoE experts accurately via phase-coherence cos(query - expert)', () => {
    const bus = new SpinWaveArenaBus({ numBands: 4 });
    const expertPhases = [
      0, // index 0
      Math.PI / 4, // index 1
      Math.PI / 2, // index 2
      (3 * Math.PI) / 4, // index 3
      Math.PI, // index 4
      -Math.PI / 2, // index 5
    ];

    // Query aligned near Math.PI / 2 + 0.05 rad
    const query = Math.PI / 2 + 0.05;
    const top3 = bus.routeTopKExperts(query, expertPhases, 3);

    assert.equal(top3.length, 3);
    // Closest is index 2 (pi/2), then index 3 (3pi/4) slightly closer than index 1 (pi/4)
    assert.equal(top3[0].index, 2);
    assert.equal(top3[1].index, 3);
    assert.equal(top3[2].index, 1);

    assert.ok(top3[0].coherence > top3[1].coherence);
    assert.ok(top3[1].coherence > top3[2].coherence);
    assert.ok(Math.abs(top3[0].coherence - Math.cos(0.05)) < 1e-12);

    // Opposite expert (index 5 = -pi/2) must have negative coherence and be excluded
    assert.ok(top3.every((r) => r.index !== 5));
    assert.ok(top3.every((r) => r.coherence > 0));
  });

  it('proves zero-copy multi-section sharing via attachExistingArena(sharedBuffer)', () => {
    const sectionA = new SpinWaveArenaBus({ numBands: 8, defaultDampingGamma: 0.5 });
    const sectionB = SpinWaveArenaBus.attachExistingArena(sectionA.buffer);
    const sectionC = attachExistingArena(sectionA.buffer);

    // Exact same underlying SharedArrayBuffer reference
    assert.strictEqual(sectionB.buffer, sectionA.buffer);
    assert.strictEqual(sectionC.buffer, sectionA.buffer);

    // Section A emits a spin wave on band 5 -> Section B immediately observes it zero-copy
    sectionA.emitWave(5, 4.0, 0);
    const bView = sectionB.sampleIntersection(5);
    assert.ok(Math.abs(bView.re - 4.0) < 1e-5);
    assert.equal(bView.epoch, 2);

    // Section B superposes a quadrature wave on band 5 -> Section C observes sqrt(4^2 + 3^2) = 5
    sectionB.emitWave(5, 3.0, Math.PI / 2);
    const cView = sectionC.sampleIntersection(5);
    assert.ok(Math.abs(cView.re - 4.0) < 1e-5);
    assert.ok(Math.abs(cView.im - 3.0) < 1e-5);
    assert.ok(Math.abs(cView.amplitude - 5.0) < 1e-5);
    assert.equal(cView.epoch, 4);

    // Section C damps the shared arena -> Section A immediately observes the damped state
    sectionC.stepDamping(2.0);
    const aAfterDamp = sectionA.sampleIntersection(5);
    assert.ok(Math.abs(aAfterDamp.amplitude - 5.0 * Math.exp(-1.0)) < 1e-4);
    assert.equal(sectionA.getGlobalEpoch(), 1);
  });

  it('rejects malformed SharedArrayBuffers on attachExistingArena', () => {
    const badMagic = new SharedArrayBuffer(HEADER_BYTES + CELL_STRIDE_BYTES);
    assert.throws(() => SpinWaveArenaBus.attachExistingArena(badMagic), /Invalid B8G_ABI_MAGIC_V3/);

    const tooSmall = new SharedArrayBuffer(16);
    assert.throws(() => SpinWaveArenaBus.attachExistingArena(tooSmall), /too small/);
  });
});
