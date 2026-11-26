import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { B8gLlmStudioSection } from './index.mts';
import { startLlmStudioServer } from './server.mts';

test('Sovereign LLM Studio unifies OpenAI /v1/*, Ollama /api/*, LoRA train/merge to CAS, and Colab GPU Host Node', async () => {
  assert.equal(B8gLlmStudioSection.kind, 'ui-surface');
  assert.equal(B8gLlmStudioSection.shape?.surface, 'canvas-2d');

  const tmpCas = mkdtempSync(join(tmpdir(), 'b8g-llm-cas-'));
  const srv = await startLlmStudioServer({ host: '127.0.0.1', port: 0, casModelsDir: tmpCas });
  try {
    // 1. UI HTML
    const html = await (await fetch(`${srv.url}/`)).text();
    assert.ok(html.includes('<!DOCTYPE html>'));

    // 2. OpenAI /v1/models & Ollama /api/tags
    const models = (await (await fetch(`${srv.url}/v1/models`)).json()) as { data: unknown[] };
    assert.ok(models.data.length >= 4);
    const tags = (await (await fetch(`${srv.url}/api/tags`)).json()) as { models: unknown[] };
    assert.ok(tags.models.length >= 4);

    // 3. OpenAI /v1/chat/completions (JSON + SSE streaming)
    const chat = (await (
      await fetch(`${srv.url}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'lumiBERT-local', messages: [{ role: 'user', content: 'EADDRINUSE on port 7430' }] }),
      })
    ).json()) as { choices: Array<{ message: { content: string } }> };
    assert.ok(chat.choices[0]!.message.content.length > 0);

    const sse = await (
      await fetch(`${srv.url}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'lumiBERT-local', stream: true, messages: [{ role: 'user', content: 'hello' }] }),
      })
    ).text();
    assert.ok(sse.includes('data: [DONE]'));

    // 4. Matryoshka /v1/embeddings
    const emb = (await (
      await fetch(`${srv.url}/v1/embeddings`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ input: ['singleton named pipe', 'os pipe host bus'], task: 'retrieval.query', dimensions: 32 }),
      })
    ).json()) as { dimensions: number; data: Array<{ embedding: number[] }>; pairwiseCosine: number };
    assert.equal(emb.dimensions, 32);
    assert.equal(emb.data[0]!.embedding.length, 32);
    assert.equal(typeof emb.pairwiseCosine, 'number');

    // 5. Real LoRA train + Merge W' = W + (alpha/r)*B*A persisted into CAS
    const tr = (await (
      await fetch(`${srv.url}/api/studio/lora/train`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ taskId: 'retrieval.query', strategy: 'direct-task-finetune', steps: 8 }),
      })
    ).json()) as { ok: boolean; steps: unknown[]; adapterSha256: string };
    assert.equal(tr.ok, true);
    assert.equal(tr.steps.length, 8);
    assert.ok(tr.adapterSha256.length >= 16);

    const merged = (await (
      await fetch(`${srv.url}/api/studio/lora/merge`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ taskId: 'retrieval.query' }),
      })
    ).json()) as { ok: boolean; persistedPath: string };
    assert.equal(merged.ok, true);
    assert.ok(existsSync(merged.persistedPath), 'merged LoRA CAS manifest must be persisted');

    // 6. Google Colab GPU Host Node registration into CloudDeviceMesh + .ipynb generation
    const colab = (await (
      await fetch(`${srv.url}/api/studio/colab/spawn`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ label: 'Colab-A100', gpuTier: 'A100-40GB', recipe: 'grpo' }),
      })
    ).json()) as { ok: boolean; node: { device: { deviceId: string } }; ipynb: { nbformat: number; cells: unknown[] } };
    assert.equal(colab.ok, true);
    assert.match(colab.node.device.deviceId, /^b8g-dev-cloud-/);
    assert.equal(colab.ipynb.nbformat, 4);
    assert.ok(colab.ipynb.cells.length >= 5);
  } finally {
    await srv.close();
    rmSync(tmpCas, { recursive: true, force: true });
  }
});
