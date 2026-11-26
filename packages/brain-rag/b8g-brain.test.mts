import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { B8gBrainServer } from './cli.mts';
import {
  appendTranscriptStep,
  loadWorkspaceProfile,
  readTranscriptSteps,
  scaffoldBrainSessionDir,
} from './profile-loader.mts';

test('B8G Universal Brain isolates tenants/sessions, indexes incrementally via CAS, and serves HTTP API', async () => {
  const work = mkdtempSync(join(tmpdir(), 'b8g-brain-test-'));
  const wsA = join(work, 'ws-a');
  const brainRoot = join(work, 'brains');
  const srv = new B8gBrainServer({ rootDir: brainRoot, port: 0 });

  try {
    const url = await srv.start();
    const { mkdirSync } = await import('node:fs');
    mkdirSync(wsA, { recursive: true });
    writeFileSync(join(wsA, 'abi.h'), '/* B8G-ABI v3 ChromeMain zero-header table */\nint ChromeMain(void);\n', 'utf8');

    // 1. Create two isolated tenants
    const tA = await fetch(`${url}/api/tenants`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tenantId: 'acme', workspaceDir: wsA }),
    });
    assert.equal(tA.status, 201);
    const tB = await fetch(`${url}/api/tenants`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tenantId: 'beta' }),
    });
    assert.equal(tB.status, 201);

    // 2. Generate fresh session for tenant acme
    const sessRes = await fetch(`${url}/api/tenants/acme/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'Architecture Sprint' }),
    });
    assert.equal(sessRes.status, 201);
    const sess = (await sessRes.json()) as { sessionId: string };
    assert.ok(sess.sessionId.startsWith('sess-'));

    // 3. Incremental CAS indexing (1st pass indexes 1 file; 2nd pass hits 100% cache!)
    const idx1 = (await (
      await fetch(`${url}/api/tenants/acme/index`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dir: wsA }),
      })
    ).json()) as { indexedFiles: number; skippedCached: number };
    assert.equal(idx1.indexedFiles, 1);

    const idx2 = (await (
      await fetch(`${url}/api/tenants/acme/index`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dir: wsA }),
      })
    ).json()) as { indexedFiles: number; skippedCached: number };
    assert.equal(idx2.indexedFiles, 0);
    assert.equal(idx2.skippedCached, 1);

    // 4. Tenant-isolated episodic memory + hybrid search
    await fetch(`${url}/api/tenants/acme/memory`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: 'Never use WSL for cross-compilation; use Zig 0.14.1', tags: ['zig', 'rule'] }),
    });

    const hitsA = (await (await fetch(`${url}/api/tenants/acme/search?q=ChromeMain+Zig`)).json()) as {
      hits: Array<{ source: string }>;
    };
    const hitsB = (await (await fetch(`${url}/api/tenants/beta/search?q=ChromeMain+Zig`)).json()) as {
      hits: Array<{ source: string }>;
    };
    assert.ok(hitsA.hits.length >= 2, 'tenant acme must retrieve both code & memory');
    assert.equal(hitsB.hits.length, 0, 'tenant beta must be 100% isolated from acme');

    // 5. Chat router (dryRun context assembly)
    const chat = (await (
      await fetch(`${url}/api/tenants/acme/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: sess.sessionId, prompt: 'How does ChromeMain work?', dryRun: true }),
      })
    ).json()) as { retrieved: unknown[]; reply: string };
    assert.ok(chat.retrieved.length >= 1);
    assert.ok(chat.reply.includes('B8G-Brain'));
  } finally {
    await srv.stop();
    rmSync(work, { recursive: true, force: true });
  }
});

test('B8G-Brain 1:1 Profile Loader reads .agents/rules + SKILL.md and scaffolds brain/<session>/transcript.jsonl', () => {
  let wsRoot = resolve(process.cwd());
  for (let i = 0; i < 6; i += 1) {
    if (existsSync(join(wsRoot, '.agents', 'rules'))) break;
    const parent = dirname(wsRoot);
    if (parent === wsRoot) break;
    wsRoot = parent;
  }
  const profile = loadWorkspaceProfile(wsRoot);
  assert.ok(profile.rules.length >= 4, 'must load workspace .agents/rules/*.md');
  assert.ok(profile.skills.length >= 2, 'must load workspace .agents/skills/*/SKILL.md');

  const tmp = mkdtempSync(join(tmpdir(), 'b8g-prof-'));
  try {
    const sDir = scaffoldBrainSessionDir(tmp, 'test-session-01');
    appendTranscriptStep(sDir, {
      step_index: 1,
      source: 'USER_EXPLICIT',
      type: 'USER_INPUT',
      status: 'DONE',
      created_at: new Date().toISOString(),
      content: 'Align the universal brain profile 1:1',
    });
    const steps = readTranscriptSteps(sDir);
    assert.equal(steps.length, 1);
    assert.equal(steps[0].step_index, 1);
    assert.equal(steps[0].type, 'USER_INPUT');
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});
