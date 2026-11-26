import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { PassThrough } from 'node:stream';
import { describe, it } from 'node:test';
import {
  SubstrateCasStore,
  SubstrateKvStore,
  Zer0qRunner,
  type Zer0qJobManifest,
} from '../zer0q-substrate/index.mts';
import {
  AgentPlatformMcpServer,
  WedgePathCollisionError,
  X402_BAAS_PROVIDERS,
  assertDisjointOwnership,
  createA2aEnvelope,
  createSwarmChannel,
  createX402PaymentChallenge,
  createX402PaymentReceipt,
  renderA2uiTree,
  validateA2uiComponent,
  verifyA2aHandoff,
  verifyX402PaymentReceipt,
} from './index.mts';

describe('packages/protocols — A2A, A2UI, x402 & MCP Server', () => {
  it('A2A createA2aEnvelope, verifyA2aHandoff, and createSwarmChannel enforce wedge (^) disjoint ownership', () => {
    assertDisjointOwnership([
      { agentId: 'worker-1', paths: ['packages/zer0q-substrate/**'] },
      { agentId: 'worker-2', paths: ['dashboards/**'] },
    ]);

    assert.throws(
      () =>
        assertDisjointOwnership([
          { agentId: 'worker-1', paths: ['packages/**'] },
          { agentId: 'worker-2', paths: ['packages/protocols/**'] },
        ]),
      WedgePathCollisionError,
    );

    const channel = createSwarmChannel({ channelId: 'swarm-4' });
    channel.registerParticipant('worker-1', ['packages/protocols/**']);
    channel.registerParticipant('worker-2', ['dashboards/**']);
    assert.throws(
      () => channel.registerParticipant('worker-3', ['packages/protocols/index.mts']),
      WedgePathCollisionError,
    );

    const env = createA2aEnvelope({
      fromAgent: 'worker-1',
      toAgent: 'worker-2',
      intent: 'handoff-manifest',
      ownedPaths: ['packages/protocols/index.mts'],
      payload: { ready: true },
    });
    assert.equal(verifyA2aHandoff(env, { expectedReceiver: 'worker-2' }).ok, true);
    assert.equal(channel.send(env).ok, true);
    const msgs = channel.drain('worker-2');
    assert.equal(msgs.length, 1);
    assert.equal((msgs[0]!.payload as { ready: boolean }).ready, true);

    const tampered = { ...env, payload: { ready: false } };
    assert.equal(verifyA2aHandoff(tampered).ok, false);
  });

  it('A2UI validates <b8g-*> custom elements and escapes HTML attributes & text', () => {
    const bad = validateA2uiComponent({ tag: 'div', attrs: { onclick: 'alert(1)' } });
    assert.equal(bad.valid, false);
    assert.ok(bad.errors.length >= 2);

    const html = renderA2uiTree({
      tag: 'b8g-section',
      attrs: { 'data-kind': 'agent-card', 'data-title': 'Xe-A "Core" & <MCP>' },
      text: '<script>alert(1)</script>',
      children: [{ tag: 'b8g-badge', attrs: { 'data-status': 'verified' }, text: 'OK' }],
    });
    assert.ok(html.startsWith('<b8g-section '));
    assert.ok(html.includes('data-title="Xe-A &quot;Core&quot; &amp; &lt;MCP&gt;"'));
    assert.ok(html.includes('&lt;script&gt;alert(1)&lt;/script&gt;'));
    assert.ok(html.includes('<b8g-badge data-status="verified">OK</b8g-badge>'));
  });

  it('x402 BaaS A2A Payments exposes all 8 providers and verifies payment challenges/receipts', () => {
    const expectedNames = [
      'Solaris',
      'Railsr',
      'Swan',
      'Modulr',
      'Stripe Connect',
      'Adyen',
      'ClearBank',
      'Marqeta',
    ];
    for (const name of expectedNames) {
      assert.ok(
        X402_BAAS_PROVIDERS.some((p) => p.name === name),
        `Missing x402 provider ${name}`,
      );
    }

    const challenge = createX402PaymentChallenge({
      amountMinor: 2500,
      currency: 'EUR',
      recipientAgent: 'agent-compute-host',
      providerId: 'swan',
      resource: 'mcp://agent.model.stamp',
    });
    assert.equal(challenge.status, 402);

    const receipt = createX402PaymentReceipt(challenge, {
      payerAgent: 'agent-xe-a',
      settlementRef: 'sepa-inst-998877',
    });
    assert.equal(verifyX402PaymentReceipt(challenge, receipt).valid, true);

    const underpaid = { ...receipt, amountMinor: 100 };
    assert.equal(verifyX402PaymentReceipt(challenge, underpaid).valid, false);
  });

  it('AgentPlatformMcpServer executes all 6 real tools via JSON-RPC 2.0 and stdio', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mcp-zer0q-'));
    try {
      const cas = new SubstrateCasStore(tmpDir);
      const kv = new SubstrateKvStore();
      const runner = new Zer0qRunner(cas, kv);
      const manifest: Zer0qJobManifest = {
        apiVersion: 'zer0q.awesome-os/v1',
        name: 'mcp-cas-test',
        inputs: { agentsMd: 'a', skills: [], workspace: 'w', prompt: 'hello' },
        execution: { model: 'lumiBERT-local', maxTokens: 100, maxWallMs: 1000 },
      };
      await runner.run(manifest, () => ({ answer: 42 }));

      const server = new AgentPlatformMcpServer({ zer0qRunner: runner });
      const listRes = await server.handleRequest({ jsonrpc: '2.0', id: 1, method: 'tools/list' });
      const tools = (listRes.result as { tools: Array<{ name: string }> }).tools;
      assert.equal(tools.length, 6);

      // 1. agent.model.catalog
      const catRes = await server.handleRequest({
        jsonrpc: '2.0',
        id: 2,
        method: 'tools/call',
        params: { name: 'agent.model.catalog', arguments: {} },
      });
      assert.ok((catRes.result as { structuredContent: { models: unknown[] } }).structuredContent.models.length >= 4);

      // 2. agent.model.stamp
      const stampRes = await server.handleRequest({
        jsonrpc: '2.0',
        id: 3,
        method: 'tools/call',
        params: { name: 'agent.model.stamp', arguments: { id: 'b8g-sovereign-moe-v1' } },
      });
      assert.equal(
        (stampRes.result as { structuredContent: { boundaryVerified: boolean } }).structuredContent.boundaryVerified,
        true,
      );

      // 3. agent.moe.verify
      const moeRes = await server.handleRequest({
        jsonrpc: '2.0',
        id: 4,
        method: 'tools/call',
        params: { name: 'agent.moe.verify', arguments: { vector: [1, 2, 3, 4], allowedScopes: ['moe.public'] } },
      });
      assert.equal((moeRes.result as { structuredContent: { ok: boolean } }).structuredContent.ok, true);

      // 4. agent.zer0q.lookup
      const zRes = await server.handleRequest({
        jsonrpc: '2.0',
        id: 5,
        method: 'tools/call',
        params: { name: 'agent.zer0q.lookup', arguments: { manifest } },
      });
      assert.equal((zRes.result as { structuredContent: { hit: boolean } }).structuredContent.hit, true);

      // 5. agent.atlaskv.query
      const kvRes = await server.handleRequest({
        jsonrpc: '2.0',
        id: 6,
        method: 'tools/call',
        params: {
          name: 'agent.atlaskv.query',
          arguments: {
            query: 'SecretVault admin key',
            triples: [
              { id: 1, head: 'AtlasKV', relation: 'uses', tail: 'HiKVP', tenantScope: 'public' },
              { id: 2, head: 'SecretVault', relation: 'admin', tail: 'key', tenantScope: 'tenant:a' },
            ],
            allowedScopes: ['public'],
          },
        },
      });
      const hits = (kvRes.result as { structuredContent: { hits: Array<{ id: number }> } }).structuredContent.hits;
      assert.ok(!hits.some((h) => h.id === 2));

      // 6. agent.hardware.fit via stdio JSON-RPC line
      const stdin = new PassThrough();
      const stdout = new PassThrough();
      const lines: string[] = [];
      stdout.on('data', (chunk: Buffer) => lines.push(chunk.toString('utf8')));
      const rl = server.serveStdio(stdin, stdout);
      stdin.write(
        `${JSON.stringify({ jsonrpc: '2.0', id: 7, method: 'tools/call', params: { name: 'agent.hardware.fit', arguments: {} } })}\n`,
      );
      await new Promise((r) => setTimeout(r, 40));
      rl.close();
      const parsedStdio = JSON.parse(lines.join('').trim()) as {
        id: number;
        result: { structuredContent: { hardware: { cpuCores: number }; recommendations: unknown[] } };
      };
      assert.equal(parsedStdio.id, 7);
      assert.ok(parsedStdio.result.structuredContent.hardware.cpuCores >= 1);
      assert.ok(parsedStdio.result.structuredContent.recommendations.length >= 1);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});
