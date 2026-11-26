/**
 * @file tutorials/02-zer0q-cas-shortcircuit.mts
 * @description Tutorial 02: ZER0Q SKIMMER Content-Addressed Short-Circuit, DEADHAND
 * Idempotent Triggers & Deterministic Replay over `packages/zer0q-substrate`.
 */

import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  DeadhandTriggerRegistry,
  SubstrateCasStore,
  SubstrateKvStore,
  Zer0qRunner,
  computeInputDigest,
  type Zer0qJobManifest,
} from '../packages/zer0q-substrate/index.mts';

async function main(): Promise<void> {
  console.log('=== Tutorial 02: ZER0Q SKIMMER CAS Short-Circuit & DEADHAND Triggers ===');

  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'zer0q-tutorial-'));
  try {
    const cas = new SubstrateCasStore(tmpRoot);
    const kv = new SubstrateKvStore();
    const runner = new Zer0qRunner(cas, kv);

    const manifest: Zer0qJobManifest = {
      apiVersion: 'zer0q.awesome-os/v1',
      name: 'boundary-audit',
      alias: 'sec-audit:latest',
      inputs: {
        agentsMd: `cas:sha512:${computeInputDigest('AGENTS.md#v1')}`,
        skills: [`cas:sha512:${computeInputDigest('SKILL:b8g-microkernel-stealify')}`],
        workspace: `uvs:merkle:${computeInputDigest('tree:main@head')}`,
        prompt: 'Audit boundary gates and emit deterministic manifest',
      },
      execution: {
        model: 'b8g-xenon-1b',
        cachePolicy: 'zero-q-lookup-or-run',
        maxTokens: 25_000,
        maxWallMs: 10_000,
      },
    };

    let executions = 0;

    // Run 1: Cold CAS Miss -> execute -> commit v11/files/<aa>/<sha512>
    const miss = await runner.run(manifest, async (ctx) => {
      executions += 1;
      ctx.recordTokens(1420, 2);
      return { summary: 'Verified all boundary gates', filesChecked: 14 };
    });
    console.log(`\nRun 1 (Cold Miss): status=${miss.status}, cached=${miss.cached}, tokens=${miss.tokensUsed}, latency=${miss.latencyMs.toFixed(2)} ms`);
    console.log(`  CAS Digest: ${miss.outputDigest.slice(0, 24)}...`);
    assert.equal(miss.status, 'COMMITTED');
    assert.equal(miss.cached, false);
    assert.equal(executions, 1);

    // Run 2: Identical state -> SKIMMER CAS Hit (< 2 ms, 0 LLM tokens, 0 executions)
    const hit = await runner.run(manifest, async () => {
      executions += 1;
      return { summary: 'Should never execute', filesChecked: 0 };
    });
    console.log(`Run 2 (Hot Hit)  : status=${hit.status}, cached=${hit.cached}, tokens=${hit.tokensUsed}, latency=${hit.latencyMs.toFixed(3)} ms`);
    assert.equal(hit.status, 'CACHE_HIT');
    assert.equal(hit.cached, true);
    assert.equal(hit.tokensUsed, 0);
    assert.equal(executions, 1);

    // Run 3: Deterministic replay & DEADHAND idempotent trigger deduplication
    const replayed = await runner.replay(miss.runId);
    assert.equal(replayed.hydrated.verified, true);

    const deadhand = new DeadhandTriggerRegistry(kv);
    deadhand.addTrigger({
      id: 'nightly-audit',
      kind: 'cron',
      scheduleOrRef: '0 2 * * *',
      manifest,
    });
    const fire1 = await deadhand.fire('nightly-audit', { tick: '2026-11-26T02:00Z' }, runner, async () => ({
      summary: 'Cron tick verified',
      filesChecked: 14,
    }));
    const fire2 = await deadhand.fire('nightly-audit', { tick: '2026-11-26T02:00Z' }, runner, async () => ({
      summary: 'Duplicate tick ignored',
      filesChecked: 0,
    }));
    console.log(`DEADHAND Trigger : fire1.dedup=${fire1.deduplicated}, fire2.dedup=${fire2.deduplicated}`);
    assert.equal(fire1.deduplicated, false);
    assert.equal(fire2.deduplicated, true);
  } finally {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
