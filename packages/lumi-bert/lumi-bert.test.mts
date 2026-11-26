/**
 * @file sections/lumi-bert/lumi-bert.test.mts
 * @description Unit tests for `sections/lumi-bert` (ModernBERT RoPE, RMSNorm,
 * GeGLU, alternating local/global attention, and unit-norm embeddings).
 */

import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import { describe, it } from 'node:test';
import {
  DEFAULT_LUMI_BERT_CONFIG,
  LUMI_BERT_CONTEXT,
  LumiBertEncoder,
  LumiBertSection,
  applyRope,
  cosineSimilarity,
  gegluFfn,
  gelu,
  rmsNorm,
  tokenizeErrorText,
} from './index.mts';

describe('sections/lumi-bert', () => {
  it('defines LumiBertSection and LUMI_BERT_CONTEXT with exact devops/arena-sab metadata', () => {
    assert.equal(LumiBertSection.id, 'b8g/lumi-bert');
    assert.equal(LumiBertSection.version, '1.0.0');
    assert.equal(LUMI_BERT_CONTEXT.boundary, 'in-process');
    assert.equal(LUMI_BERT_CONTEXT.display, 'headless');
    assert.equal(LUMI_BERT_CONTEXT.lifetime, 'oneshot');
    assert.equal(LUMI_BERT_CONTEXT.transport, 'arena-sab');
    assert.equal(LUMI_BERT_CONTEXT.locality, 'local-arena');
    assert.equal(LUMI_BERT_CONTEXT.persona, 'devops');
    assert.equal(LUMI_BERT_CONTEXT.capability.tenantId, 'local-user');
    assert.equal(LUMI_BERT_CONTEXT.capability.epoch, 1);
    assert.equal(LUMI_BERT_CONTEXT.capability.privateData, true);
    assert.deepEqual(LUMI_BERT_CONTEXT.capability.scopes, ['lumibert.embed']);
  });

  it('tokenizeErrorText preserves diagnostic codes and paths without collapsing to <unk>', () => {
    const tokens = tokenizeErrorText(
      'Error: listen EADDRINUSE 127.0.0.1:8890 in third_party/rust/read_fonts B8G018',
    );
    assert.ok(tokens.length > 5);
    for (const id of tokens) {
      assert.ok(id >= 1 && id < DEFAULT_LUMI_BERT_CONFIG.vocabSize, `unexpected token id ${id}`);
    }

    const emptyTokens = tokenizeErrorText('   ');
    assert.deepEqual(emptyTokens, [1]);
  });

  it('rmsNorm normalizes vector to unit RMS', () => {
    const input = new Float32Array([3, -4, 5, -6]);
    const normed = rmsNorm(input);
    let meanSq = 0;
    for (let i = 0; i < normed.length; i += 1) {
      meanSq += normed[i]! * normed[i]!;
    }
    meanSq /= normed.length;
    assert.ok(Math.abs(Math.sqrt(meanSq) - 1.0) < 1e-4, `expected unit RMS, got ${Math.sqrt(meanSq)}`);
  });

  it('applyRope preserves L2 norm, is identity at pos=0, and distinguishes local vs global theta', () => {
    const vec = new Float32Array([1.5, -0.5, 0.25, 2.0, -1.0, 0.75, 0.5, -1.25]);
    const origNorm = Math.hypot(...vec);

    const pos0 = applyRope(vec, 0, 10000);
    for (let i = 0; i < vec.length; i += 1) {
      assert.ok(Math.abs(pos0[i]! - vec[i]!) < 1e-6);
    }

    const localRot = applyRope(vec, 7, DEFAULT_LUMI_BERT_CONFIG.localRopeTheta);
    const globalRot = applyRope(vec, 7, DEFAULT_LUMI_BERT_CONFIG.globalRopeTheta);

    assert.ok(Math.abs(Math.hypot(...localRot) - origNorm) < 1e-5);
    assert.ok(Math.abs(Math.hypot(...globalRot) - origNorm) < 1e-5);

    // Local theta (10000) vs Global theta (160000) must produce distinct rotations at higher dims
    let maxDiff = 0;
    for (let i = 0; i < vec.length; i += 1) {
      maxDiff = Math.max(maxDiff, Math.abs(localRot[i]! - globalRot[i]!));
    }
    assert.ok(maxDiff > 1e-3, `expected local vs global RoPE difference, got ${maxDiff}`);
  });

  it('gegluFfn computes exact GELU(gate) * up followed by down projection', () => {
    const dim = 2;
    const ffnDim = 2;
    const x = new Float32Array([1.0, -2.0]);
    const wGate = new Float32Array([1, 0, 0, 1]); // gate = [1, -2]
    const wUp = new Float32Array([2, 0, 0, 3]); // up = [2, -6]
    const wDown = new Float32Array([1, 0, 0, 1]); // identity down

    const out = gegluFfn(x, wGate, wUp, wDown, dim, ffnDim);
    const expected0 = gelu(1.0) * 2.0;
    const expected1 = gelu(-2.0) * -6.0;

    assert.ok(Math.abs(out[0]! - expected0) < 1e-5);
    assert.ok(Math.abs(out[1]! - expected1) < 1e-5);
  });

  it('LumiBertEncoder produces deterministic unit-norm embeddings with higher similarity for same error class', () => {
    const encoder = new LumiBertEncoder();
    assert.equal(encoder.isPretrainedWeightsLoaded, false);

    const e1 = encoder.embed('Error: listen EADDRINUSE: address already in use 127.0.0.1:8890');
    const e2 = encoder.embed('Error: listen EADDRINUSE: address already in use 0.0.0.0:3000');
    const e3 = encoder.embed('ERROR at //.gn:18:21: This is not a string. script_executable = ["python3"]');

    assert.equal(e1.length, encoder.config.dim);
    const norm1 = Math.hypot(...e1);
    assert.ok(Math.abs(norm1 - 1.0) < 1e-5, `expected unit norm, got ${norm1}`);

    // Determinism check
    const e1Repeat = encoder.embed('Error: listen EADDRINUSE: address already in use 127.0.0.1:8890');
    assert.ok(cosineSimilarity(e1, e1Repeat) > 0.999999);

    const sameClassSim = encoder.cosineSimilarity(e1, e2);
    const diffClassSim = encoder.cosineSimilarity(e1, e3);
    assert.ok(
      sameClassSim > diffClassSim + 0.25,
      `expected sameClassSim (${sameClassSim.toFixed(4)}) > diffClassSim (${diffClassSim.toFixed(4)}) + 0.25`,
    );

    // Latency check (< 5 ms average after warmup)
    const iters = 15;
    const t0 = performance.now();
    for (let i = 0; i < iters; i += 1) {
      encoder.embed('B8G018 HTML table column mismatch tr td th count');
    }
    const avgMs = (performance.now() - t0) / iters;
    assert.ok(avgMs < 5.0, `expected < 5 ms per embed, got ${avgMs.toFixed(3)} ms`);
  });
});
