import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { describe, it } from 'node:test';
import {
  DeadhandTriggerRegistry,
  SubstrateCasStore,
  SubstrateKvStore,
  Zer0qRunner,
  canonicalJson,
  computeInputDigest,
  type Zer0qJobManifest,
} from './index.mts';

describe('packages/zer0q-substrate — CAS, KV, SKIMMER, Replay & Deadhand', () => {
  it('canonicalJson & computeInputDigest produce deterministic SHA-512 independent of key order', () => {
    const a = { z: 1, a: { c: 3, b: 2 }, u: undefined, arr: [1, { y: 2, x: 1 }] };
    const b = { arr: [1, { x: 1, y: 2 }], a: { b: 2, c: 3 }, z: 1 };
    assert.equal(canonicalJson(a), canonicalJson(b));
    assert.equal(computeInputDigest(a), computeInputDigest(b));
    assert.equal(computeInputDigest(a).length, 128);
  });

  it('SubstrateCasStore persists under v11/files/<aa>/<sha512>, verifies integrity, pins, and GCs', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'zer0q-cas-'));
    try {
      const cas = new SubstrateCasStore(tmpDir);
      const p1 = cas.put({ hello: 'zer0q' });
      const p2 = cas.put({ hello: 'zer0q' });
      assert.equal(p1.sha512, p2.sha512);
      assert.equal(p2.deduplicated, true);
      assert.ok(p1.path.includes(path.join('v11', 'files', p1.sha512.slice(0, 2), p1.sha512)));
      assert.equal(cas.has(p1.sha512), true);
      assert.equal(cas.verify(p1.sha512), true);
      assert.deepEqual(cas.getJson(p1.sha512), { hello: 'zer0q' });

      const unpinned = cas.put('ephemeral-artifact');
      cas.pin(p1.sha512, 'keep');
      assert.equal(cas.isPinned(p1.sha512), true);
      const gcRes = cas.gc();
      assert.equal(gcRes.removed, 1);
      assert.equal(gcRes.retained, 1);
      assert.equal(cas.has(unpinned.sha512), false);
      assert.equal(cas.has(p1.sha512), true);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('SubstrateKvStore supports atomic compareAndSwap, TTL leases, idempotency & budgets', () => {
    const kv = new SubstrateKvStore();
    assert.equal(kv.compareAndSwap('alias:prod', null, 'sha-v1'), true);
    assert.equal(kv.compareAndSwap('alias:prod', null, 'sha-v2'), false);
    assert.equal(kv.compareAndSwap('alias:prod', 'sha-v1', 'sha-v2'), true);
    assert.equal(kv.getAlias('alias:prod'), 'sha-v2');

    const lease1 = kv.acquireLease('run-1', { owner: 'worker-a', ttlMs: 100, nowMs: 1000 });
    assert.ok(lease1);
    assert.equal(kv.acquireLease('run-1', { owner: 'worker-b', ttlMs: 100, nowMs: 1050 }), null);
    assert.ok(kv.renewLease('run-1', 'worker-a', 200, 1080));
    const reapedNone = kv.reapExpiredLeases(1200);
    assert.equal(reapedNone.length, 0);
    const reapedOne = kv.reapExpiredLeases(1300);
    assert.equal(reapedOne.length, 1);

    kv.initBudget('run-1', { maxTokens: 100, maxWallMs: 500, maxToolCalls: 2 }, 1000);
    const tick1 = kv.recordBudgetTick('run-1', { tokensUsed: 60, toolCalls: 1 }, 1100);
    assert.equal(tick1.ok, true);
    assert.equal(tick1.remainingTokens, 40);
    const tick2 = kv.recordBudgetTick('run-1', { tokensUsed: 50, toolCalls: 1 }, 1200);
    assert.equal(tick2.ok, false);
    assert.equal(tick2.exceededReason, 'TOKEN_CEILING_EXCEEDED');
  });

  it('Zer0qRunner SKIMMER short-circuits identical runs in < 2 ms with 0 tokens and supports replay()', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'zer0q-run-'));
    try {
      const cas = new SubstrateCasStore(tmpDir);
      const kv = new SubstrateKvStore();
      const runner = new Zer0qRunner(cas, kv);

      const manifest: Zer0qJobManifest = {
        apiVersion: 'zer0q.awesome-os/v1',
        name: 'audit-auth',
        alias: 'sec-audit:latest',
        inputs: {
          agentsMd: 'cas:sha512:agents',
          skills: ['cas:sha512:skill-sec'],
          workspace: 'uvs:merkle:root1',
          prompt: 'Verify auth boundary invariants',
        },
        execution: {
          model: 'gemini-2.5-pro',
          maxTokens: 50_000,
          maxWallMs: 30_000,
        },
      };

      let executions = 0;
      const run1 = await runner.run(manifest, (ctx) => {
        executions += 1;
        ctx.recordTokens(420, 2);
        return { output: { patch: 'ok', issues: 0 }, tokensUsed: 420 };
      });
      assert.equal(run1.status, 'COMMITTED');
      assert.equal(run1.cached, false);
      assert.equal(run1.tokensUsed, 420);
      assert.equal(executions, 1);

      const lookupHit = runner.lookup(manifest);
      assert.equal(lookupHit.hit, true);
      assert.equal(lookupHit.tokensUsed, 0);
      assert.ok(lookupHit.latencyMs < 2.0, `expected < 2 ms SKIMMER lookup, got ${lookupHit.latencyMs} ms`);

      const run2 = await runner.run(manifest, () => {
        executions += 1;
        return { patch: 'should-not-run' };
      });
      assert.equal(run2.status, 'CACHE_HIT');
      assert.equal(run2.cached, true);
      assert.equal(run2.tokensUsed, 0);
      assert.equal(executions, 1);
      assert.ok(run2.latencyMs < 2.0, `expected < 2 ms SKIMMER run short-circuit, got ${run2.latencyMs} ms`);

      const replayRes = await runner.replay(run1.runId, {
        modelOverride: 'gemini-3-pro',
        executor: (snap) => ({ replayedFrom: snap.runId, patched: true }),
      });
      assert.equal(replayRes.hydrated.verified, true);
      assert.equal(replayRes.reexecuted, true);
      assert.equal(replayRes.modelUsed, 'gemini-3-pro');
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('DeadhandTriggerRegistry deduplicates identical trigger payloads via KV idempotency keys', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'zer0q-deadhand-'));
    try {
      const cas = new SubstrateCasStore(tmpDir);
      const kv = new SubstrateKvStore();
      const runner = new Zer0qRunner(cas, kv);
      const registry = new DeadhandTriggerRegistry(kv);

      registry.addTrigger({
        id: 'nightly-audit',
        kind: 'cron',
        scheduleOrRef: '0 2 * * *',
        manifest: {
          apiVersion: 'zer0q.awesome-os/v1',
          name: 'nightly-audit',
          inputs: {
            agentsMd: 'agents',
            skills: [],
            workspace: 'tree-1',
            prompt: 'Run nightly check',
          },
          execution: { model: 'lumiBERT-local', maxTokens: 1000, maxWallMs: 5000 },
        },
      });

      const first = await registry.fire('nightly-audit', { tick: '2026-11-26T02:00:00Z' }, runner, () => ({
        ok: true,
      }));
      assert.equal(first.deduplicated, false);
      assert.equal(first.runResult?.status, 'COMMITTED');

      const duplicate = await registry.fire('nightly-audit', { tick: '2026-11-26T02:00:00Z' }, runner, () => ({
        ok: false,
      }));
      assert.equal(duplicate.deduplicated, true);
      assert.equal(duplicate.runResult, undefined);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});
