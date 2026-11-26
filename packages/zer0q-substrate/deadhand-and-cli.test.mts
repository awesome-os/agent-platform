import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { runZer0qCli } from './cli.mts';
import { DeadhandController, buildIdempotencyKey } from './deadhand.mts';
import {
  SubstrateCasStore,
  SubstrateKvStore,
  Zer0qRunner,
  type Zer0qJobManifest,
} from './index.mts';
import { assertNoSecrets, redactString, redactValue } from './secret-redactor.mts';

describe('packages/zer0q-substrate — Secret Redactor, DeadhandController & CLI', () => {
  it('secret-redactor strips API keys, Bearer tokens, ghp_/github_pat_, TWITCH_STREAM_KEY, and PEM blocks', () => {
    const fakeGithubClassic = 'ghp_1234567890abcdefghijklmnop';
    const fakeGithubPat = 'github_pat_11ABCDEFG0123456789_abcdef1234567890';
    const fakeBearer = 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig';
    const fakeTwitchKey = 'live_987654321_abcdefABCDEF1234567890';
    const fakeOpenAiKey = 'sk-proj-abcdefghijklmnopqrstuvwxyz123456';
    const fakePem = [
      '-----BEGIN RSA PRIVATE KEY-----',
      'MIIEowIBAAKCAQEAnFakeKeyMaterial0123456789',
      '-----END RSA PRIVATE KEY-----',
    ].join('\n');

    const rawLog = [
      `Authorization: ${fakeBearer}`,
      `GH=${fakeGithubClassic} PAT=${fakeGithubPat}`,
      `TWITCH_STREAM_KEY=${fakeTwitchKey}`,
      `OPENAI=${fakeOpenAiKey}`,
      fakePem,
    ].join('\n');

    const strRes = redactString(rawLog);
    assert.ok(strRes.count >= 5);
    for (const secret of [
      'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9',
      fakeGithubClassic,
      fakeGithubPat,
      fakeTwitchKey,
      fakeOpenAiKey,
      'MIIEowIBAAKCAQEAnFakeKeyMaterial0123456789',
    ]) {
      assert.equal(strRes.redacted.includes(secret), false, `secret leaked: ${secret}`);
    }
    assertNoSecrets(strRes.redacted);

    const objRes = redactValue({
      apiKey: 'secret-value-12345678',
      nested: { TWITCH_STREAM_KEY: fakeTwitchKey, note: `token=${fakeGithubClassic}` },
    });
    assertNoSecrets(objRes.redacted);
  });

  it('Zer0qRunner scrubs secrets from manifests, outputs, and replays before CAS commit', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'zer0q-redact-cas-'));
    try {
      const cas = new SubstrateCasStore(tmpDir);
      const kv = new SubstrateKvStore();
      const runner = new Zer0qRunner(cas, kv);

      const fakeGithubClassic = 'ghp_1234567890abcdefghijklmnop';
      const fakeTwitchKey = 'live_987654321_abcdefABCDEF1234567890';
      const manifest: Zer0qJobManifest = {
        apiVersion: 'zer0q.awesome-os/v1',
        name: 'secret-scrub-check',
        inputs: {
          agentsMd: 'AGENTS.md',
          skills: [],
          workspace: 'uvs:tree',
          prompt: `Inspect ${fakeGithubClassic}`,
        },
        execution: { model: 'b8g-xenon-1b', maxTokens: 5000, maxWallMs: 5000 },
      };

      const runRes = await runner.run(manifest, () => ({
        output: {
          transcript: `TWITCH_STREAM_KEY=${fakeTwitchKey}`,
          api_key: 'super-secret-api-key-999',
        },
        tokensUsed: 42,
      }));

      const rawDiskBlob = fs.readFileSync(cas.blobPath(runRes.outputDigest), 'utf8');
      for (const secret of [fakeGithubClassic, fakeTwitchKey, 'super-secret-api-key-999']) {
        assert.equal(rawDiskBlob.includes(secret), false, `CAS blob leaked secret: ${secret}`);
      }
      assertNoSecrets(JSON.parse(rawDiskBlob));

      const replayRes = await runner.replay(runRes.runId, {
        modelOverride: 'gemini-3-pro',
        executor: () => ({ note: `Bearer token-secret-abcdefghijklmnopqrstuvwxyz` }),
      });
      const rawReplayBlob = fs.readFileSync(cas.blobPath(replayRes.outputDigest), 'utf8');
      assert.equal(rawReplayBlob.includes('token-secret-abcdefghijklmnopqrstuvwxyz'), false);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('DeadhandController SKIMMER short-circuits identical runs in < 2 ms with 0 tokens', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'zer0q-skimmer-2ms-'));
    try {
      const cas = new SubstrateCasStore(tmpDir);
      const kv = new SubstrateKvStore();
      const runner = new Zer0qRunner(cas, kv);
      const manifest: Zer0qJobManifest = {
        apiVersion: 'zer0q.awesome-os/v1',
        name: 'fast-skimmer',
        inputs: {
          agentsMd: 'AGENTS.md',
          skills: ['zer0q'],
          workspace: 'tree:v1',
          prompt: 'Verify invariants',
        },
        execution: { model: 'gemini-2.5-pro', maxTokens: 10_000, maxWallMs: 10_000 },
      };

      const firstRun = await runner.run(manifest, (ctx) => {
        ctx.recordTokens(250, 1);
        return { output: { ok: true }, tokensUsed: 250 };
      });
      assert.equal(firstRun.status, 'COMMITTED');

      const shortCircuit = await runner.run(manifest, () => {
        throw new Error('Executor must not run on SKIMMER hit');
      });
      assert.equal(shortCircuit.status, 'CACHE_HIT');
      assert.equal(shortCircuit.tokensUsed, 0);
      assert.ok(shortCircuit.latencyMs < 2.0, `Expected < 2 ms short-circuit, got ${shortCircuit.latencyMs} ms`);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('DeadhandController deduplicates cron and webhook triggers via idem:<trigger>:<digest>', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'zer0q-triggers-'));
    try {
      const cas = new SubstrateCasStore(tmpDir);
      const deadhand = new DeadhandController({ cas });
      const manifest: Zer0qJobManifest = {
        apiVersion: 'zer0q.awesome-os/v1',
        name: 'trigger-job',
        inputs: { agentsMd: 'AGENTS.md', skills: [], workspace: 'tree', prompt: 'Check' },
        execution: { model: 'b8g-xenon-1b', maxTokens: 2000, maxWallMs: 2000 },
      };

      deadhand.registerTrigger({ id: 'cron-audit', kind: 'cron', scheduleOrRef: '0 3 * * *', manifest });
      const cron1 = await deadhand.fireCron('cron-audit', '2026-11-26T03:00:00Z', () => ({ ok: true }));
      const cron2 = await deadhand.fireCron('cron-audit', '2026-11-26T03:00:00Z', () => ({ ok: false }));
      assert.equal(cron1.deduplicated, false);
      assert.equal(cron2.deduplicated, true);
      assert.ok(cron1.idempotencyKey.startsWith('idem:cron-audit:'));

      deadhand.registerTrigger({ id: 'gh-pr-hook', kind: 'webhook', scheduleOrRef: 'pull_request.opened', manifest });
      const wh1 = await deadhand.fireWebhook('gh-pr-hook', 'deliv-101', { pr: 42 }, () => ({ ok: true }));
      const wh2 = await deadhand.fireWebhook('gh-pr-hook', 'deliv-101', { pr: 42 }, () => ({ ok: false }));
      assert.equal(wh1.deduplicated, false);
      assert.equal(wh2.deduplicated, true);
      assert.equal(buildIdempotencyKey('x', { b: 2, a: 1 }), buildIdempotencyKey('x', { a: 1, b: 2 }));
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('DeadhandController cas-diff skips unchanged trees and deduplicates identical diffs', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'zer0q-casdiff-'));
    try {
      const cas = new SubstrateCasStore(tmpDir);
      const deadhand = new DeadhandController({ cas });
      const manifest: Zer0qJobManifest = {
        apiVersion: 'zer0q.awesome-os/v1',
        name: 'diff-job',
        inputs: { agentsMd: 'AGENTS.md', skills: [], workspace: 'tree', prompt: 'Diff' },
        execution: { model: 'b8g-xenon-1b', maxTokens: 2000, maxWallMs: 2000 },
      };

      deadhand.registerTrigger({ id: 'tree-diff', kind: 'cas-diff', scheduleOrRef: 'refs/heads/main', manifest });
      const unchanged = await deadhand.fireCasDiff(
        'tree-diff',
        { prevTreeDigest: 'aaa', nextTreeDigest: 'aaa' },
        () => ({ ok: true }),
      );
      assert.equal(unchanged.skippedUnchanged, true);

      const diff1 = await deadhand.fireCasDiff(
        'tree-diff',
        { prevTreeDigest: 'aaa', nextTreeDigest: 'bbb', changedFiles: ['src/index.mts'] },
        () => ({ ok: true }),
      );
      const diff2 = await deadhand.fireCasDiff(
        'tree-diff',
        { prevTreeDigest: 'aaa', nextTreeDigest: 'bbb', changedFiles: ['src/index.mts'] },
        () => ({ ok: false }),
      );
      assert.equal(diff1.skippedUnchanged, false);
      assert.equal(diff1.fireResult?.deduplicated, false);
      assert.equal(diff2.fireResult?.deduplicated, true);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('DeadhandController supports atomic alias set/get/rollback revision stack', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'zer0q-alias-'));
    try {
      const cas = new SubstrateCasStore(tmpDir);
      const deadhand = new DeadhandController({ cas });

      assert.equal(deadhand.setAlias('release:current', 'sha-v1').ok, true);
      assert.equal(deadhand.setAlias('release:current', 'sha-v2').ok, true);
      assert.equal(deadhand.setAlias('release:current', 'sha-v3').ok, true);
      assert.equal(deadhand.getAlias('release:current'), 'sha-v3');
      assert.deepEqual(deadhand.getAliasHistory('release:current'), ['sha-v1', 'sha-v2']);

      const rb1 = deadhand.rollbackAlias('release:current');
      assert.equal(rb1.ok, true);
      assert.equal(rb1.currentHash, 'sha-v2');
      const rb2 = deadhand.rollbackAlias('release:current');
      assert.equal(rb2.ok, true);
      assert.equal(rb2.currentHash, 'sha-v1');
      assert.equal(deadhand.rollbackAlias('release:current').ok, false);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('DeadhandController tracks TTL lease heartbeats and reaps expired orphan leases', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'zer0q-leases-'));
    try {
      const cas = new SubstrateCasStore(tmpDir);
      const deadhand = new DeadhandController({ cas });

      const l1 = deadhand.acquireLease('run-live', { owner: 'worker-1', ttlMs: 100, nowMs: 1000 });
      const l2 = deadhand.acquireLease('run-orphan', { owner: 'worker-dead', ttlMs: 50, nowMs: 1000 });
      assert.ok(l1 && l2);
      assert.ok(deadhand.heartbeatLease('run-live', 'worker-1', 200, 1060));

      const reaped = deadhand.reapOrphanLeases(1100);
      assert.equal(reaped.length, 1);
      assert.equal(reaped[0]?.runId, 'run-orphan');
      assert.equal(deadhand.listActiveLeases(1100).length, 1);
      assert.equal(deadhand.releaseLease('run-live', 'worker-1'), true);
      assert.equal(deadhand.listActiveLeases(1100).length, 0);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('DeadhandController executes deterministic replay(runId, overrideModel)', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'zer0q-replay-'));
    try {
      const cas = new SubstrateCasStore(tmpDir);
      const deadhand = new DeadhandController({ cas });
      const manifest: Zer0qJobManifest = {
        apiVersion: 'zer0q.awesome-os/v1',
        name: 'replay-job',
        inputs: { agentsMd: 'AGENTS.md', skills: ['zer0q'], workspace: 'tree', prompt: 'Replay me' },
        execution: { model: 'gemini-2.5-pro', maxTokens: 5000, maxWallMs: 5000 },
      };

      const first = await deadhand.runner.run(manifest, () => ({ answer: 'v1' }));
      const replayed = await deadhand.replay(first.runId, 'gemini-3-pro', (snap) => ({
        replayedRun: snap.runId,
        verified: snap.verified,
        prev: snap.originalOutput,
      }));
      assert.equal(replayed.hydrated.verified, true);
      assert.equal(replayed.reexecuted, true);
      assert.equal(replayed.modelUsed, 'gemini-3-pro');
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('cli.mts supports run, lookup, alias, replay, --json, and --help', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'zer0q-cli-core-'));
    try {
      const casDir = path.join(tmpDir, 'cas');
      const manifestPath = path.join(tmpDir, 'job.json');
      const manifest: Zer0qJobManifest = {
        apiVersion: 'zer0q.awesome-os/v1',
        name: 'cli-smoke-job',
        alias: 'cli:latest',
        inputs: { agentsMd: 'AGENTS.md', skills: ['zer0q'], workspace: 'tree:cli', prompt: 'Verify CLI' },
        execution: { model: 'b8g-xenon-1b', maxTokens: 5000, maxWallMs: 5000 },
      };
      fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), 'utf8');

      const invoke = async (args: string[]) => {
        const out: string[] = [];
        const err: string[] = [];
        const code = await runZer0qCli([...args, '--cas', casDir, '--json'], {
          stdout: (l) => out.push(l),
          stderr: (l) => err.push(l),
        });
        return {
          code,
          json: out.length > 0 ? (JSON.parse(out.join('\n')) as Record<string, unknown>) : null,
        };
      };

      assert.equal((await invoke(['--help'])).code, 0);
      assert.equal((await invoke(['lookup', manifestPath])).json?.hit, false);

      const run1 = await invoke(['run', manifestPath]);
      assert.equal(run1.code, 0);
      assert.equal(run1.json?.status, 'COMMITTED');
      const runId = String(run1.json?.runId);

      const hit = await invoke(['lookup', manifestPath]);
      assert.equal(hit.json?.hit, true);
      assert.equal(hit.json?.tokensUsed, 0);

      const run2 = await invoke(['run', manifestPath]);
      assert.equal(run2.json?.status, 'CACHE_HIT');

      assert.equal((await invoke(['alias', 'set', 'deploy:prod', 'digest-1'])).code, 0);
      assert.equal((await invoke(['alias', 'set', 'deploy:prod', 'digest-2'])).code, 0);
      assert.equal((await invoke(['alias', 'get', 'deploy:prod'])).json?.casHash, 'digest-2');
      assert.equal((await invoke(['alias', 'rollback', 'deploy:prod'])).json?.currentHash, 'digest-1');
      assert.equal((await invoke(['alias', 'list'])).code, 0);

      const replayRes = await invoke(['replay', runId, '--model', 'gemini-3-pro']);
      assert.equal(replayRes.code, 0);
      assert.equal(replayRes.json?.modelUsed, 'gemini-3-pro');
      assert.equal(replayRes.json?.reexecuted, true);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('cli.mts supports trigger, leases, and direct Node subprocess execution', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'zer0q-cli-ops-'));
    try {
      const casDir = path.join(tmpDir, 'cas');
      const manifestPath = path.join(tmpDir, 'job.json');
      const manifest: Zer0qJobManifest = {
        apiVersion: 'zer0q.awesome-os/v1',
        name: 'cli-ops-job',
        inputs: { agentsMd: 'AGENTS.md', skills: [], workspace: 'tree', prompt: 'Verify ops' },
        execution: { model: 'b8g-xenon-1b', maxTokens: 5000, maxWallMs: 5000 },
      };
      fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), 'utf8');

      const invoke = async (args: string[]) => {
        const out: string[] = [];
        const code = await runZer0qCli([...args, '--cas', casDir, '--json'], {
          stdout: (l) => out.push(l),
          stderr: () => undefined,
        });
        return {
          code,
          json: out.length > 0 ? (JSON.parse(out.join('\n')) as Record<string, unknown>) : null,
        };
      };

      const trigAdd = await invoke([
        'trigger',
        'add',
        'nightly',
        '--kind',
        'cron',
        '--ref',
        '0 2 * * *',
        '--manifest',
        manifestPath,
      ]);
      assert.equal(trigAdd.code, 0);
      assert.equal(((await invoke(['trigger', 'list'])).json?.triggers as unknown[]).length, 1);
      const trigFire = await invoke(['trigger', 'fire', 'nightly', '--payload', '{"tick":"t1"}']);
      assert.equal(trigFire.code, 0);
      assert.equal(trigFire.json?.deduplicated, false);

      assert.equal(
        (await invoke(['leases', 'acquire', 'lease-1', '--owner', 'w1', '--ttl', '50', '--now', '1000'])).code,
        0,
      );
      assert.equal(((await invoke(['leases', 'list', '--now', '1020'])).json?.leases as unknown[]).length, 1);
      assert.equal(
        (await invoke(['leases', 'heartbeat', 'lease-1', '--owner', 'w1', '--ttl', '100', '--now', '1040'])).code,
        0,
      );
      assert.equal(((await invoke(['leases', 'reap', '--now', '1100'])).json?.reaped as unknown[]).length, 0);
      assert.equal(((await invoke(['leases', 'reap', '--now', '1200'])).json?.reaped as unknown[]).length, 1);

      const cliFile = path.join(path.dirname(fileURLToPath(import.meta.url)), 'cli.mts');
      const rawOut = execFileSync(process.execPath, [cliFile, '--help', '--json'], { encoding: 'utf8' });
      assert.equal((JSON.parse(rawOut) as { readonly ok: boolean }).ok, true);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});
