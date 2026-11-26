#!/usr/bin/env node
/**
 * @file server.mts
 * @description Unified OpenAI (`/v1/*`) + Ollama (`/api/*`) + LM Studio + Unsloth LoRA
 * + Google Colab Host Node Server (`.\bin\b8g-llm-studio`, default port `7497`).
 */
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { CloudDeviceMesh } from './cloud-devices.mts';
import {
  SovereignLlmStudio,
  UNSLOTH_RECIPES,
  buildLlamaCppLaunchArgs,
  buildUnslothExportPlan,
  parseOllamaModelfile,
  searchHuggingFaceGguf,
  validateDataset,
  type DatasetFormat,
  type UnslothExportConfig,
} from './index.mts';
import type { ContrastivePair, LoraStrategy, QuantizationKind, TaskAdapterId } from './lora-engine.mts';
import type { ColabGpuTier, GenerateColabNotebookOptions } from './colab-node-bridge.mts';
import { renderLlmStudioHtml } from './ui-html.mts';

export async function startLlmStudioServer(opts: {
  host?: string;
  port?: number;
  casModelsDir?: string;
  deviceMesh?: CloudDeviceMesh;
} = {}): Promise<{ url: string; studio: SovereignLlmStudio; close: () => Promise<void> }> {
  const host = opts.host ?? '127.0.0.1';
  const port = opts.port ?? 7497;
  const studio = new SovereignLlmStudio({
    casModelsDir: opts.casModelsDir,
    deviceMesh: opts.deviceMesh,
  });

  const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
    try {
      res.setHeader('access-control-allow-origin', '*');
      res.setHeader('access-control-allow-headers', 'content-type, authorization');
      res.setHeader('access-control-allow-methods', 'GET, POST, OPTIONS');
      if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
      }

      const url = new URL(req.url ?? '/', `http://${host}:${port}`);

      if (req.method === 'GET' && url.pathname === '/') {
        res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
        res.end(renderLlmStudioHtml());
        return;
      }

      // OpenAI-compatible GET /v1/models
      if (req.method === 'GET' && url.pathname === '/v1/models') {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify(studio.hub.toOpenAiModelsResponse()));
        return;
      }

      // Ollama-compatible GET /api/tags
      if (req.method === 'GET' && url.pathname === '/api/tags') {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify(studio.hub.toOllamaTagsResponse()));
        return;
      }

      // Studio Hub HF GGUF Search
      if (req.method === 'GET' && url.pathname === '/api/studio/hub/search') {
        const q = url.searchParams.get('q') ?? 'qwen';
        const limit = Number(url.searchParams.get('limit') ?? 10);
        const result = await searchHuggingFaceGguf(q, { limit });
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify(result));
        return;
      }

      // Studio LoRA adapters & Unsloth recipes overview
      if (req.method === 'GET' && url.pathname === '/api/studio/lora/adapters') {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(
          JSON.stringify({
            adapters: studio.lora.listAdapters(),
            recipes: UNSLOTH_RECIPES,
            colabNodes: studio.colab.listNodes(),
          }),
        );
        return;
      }

      const chunks: Buffer[] = [];
      for await (const c of req) chunks.push(typeof c === 'string' ? Buffer.from(c) : c);
      const rawBody = Buffer.concat(chunks).toString('utf8');
      const body = rawBody ? (JSON.parse(rawBody) as Record<string, unknown>) : {};

      // OpenAI /v1/chat/completions & Ollama /api/chat (SSE streaming + JSON)
      if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/api/chat')) {
        const messages = Array.isArray(body.messages)
          ? (body.messages as Array<{ role: string; content: string }>)
          : [];
        const lastUser = [...messages].reverse().find((m) => m.role === 'user')?.content ?? String(body.prompt ?? '');
        const model = String(body.model ?? 'lumiBERT-local');
        const wantStream = Boolean(body.stream);
        const reply = studio.ai.prompt(lastUser);

        if (wantStream && url.pathname === '/v1/chat/completions') {
          res.writeHead(200, {
            'content-type': 'text/event-stream; charset=utf-8',
            'cache-control': 'no-cache',
          });
          const tokenStream = studio.ai.promptStreaming(lastUser);
          const reader = tokenStream.getReader();
          const id = `chatcmpl-${Date.now()}`;
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const chunk = {
              id,
              object: 'chat.completion.chunk',
              created: Math.floor(Date.now() / 1000),
              model,
              choices: [{ index: 0, delta: { content: value }, finish_reason: null }],
            };
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          }
          res.write('data: [DONE]\n\n');
          res.end();
          return;
        }

        res.writeHead(200, { 'content-type': 'application/json' });
        if (url.pathname === '/api/chat') {
          res.end(
            JSON.stringify({
              model,
              created_at: new Date().toISOString(),
              message: { role: 'assistant', content: reply.text },
              done: true,
            }),
          );
        } else {
          res.end(
            JSON.stringify({
              id: `chatcmpl-${Date.now()}`,
              object: 'chat.completion',
              created: Math.floor(Date.now() / 1000),
              model,
              choices: [
                {
                  index: 0,
                  message: { role: 'assistant', content: reply.text },
                  finish_reason: 'stop',
                },
              ],
              llamaArgs: buildLlamaCppLaunchArgs({
                modelPath: `${model}.gguf`,
                port: 8080,
                ctxSize: Number(body.ctxSize ?? 4096),
                gpuLayers: Number(body.gpuLayers ?? 35),
              }),
            }),
          );
        }
        return;
      }

      // OpenAI /v1/embeddings & Ollama /api/embeddings
      if (req.method === 'POST' && (url.pathname === '/v1/embeddings' || url.pathname === '/api/embeddings')) {
        const rawInput = body.input ?? body.prompt ?? '';
        const inputs = Array.isArray(rawInput) ? rawInput.map(String) : [String(rawInput)];
        const out = studio.embedTexts({
          inputs,
          model: typeof body.model === 'string' ? body.model : undefined,
          task: (typeof body.task === 'string' ? body.task : 'retrieval.query') as TaskAdapterId,
          dimensions: typeof body.dimensions === 'number' ? body.dimensions : 64,
        });
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify(out));
        return;
      }

      // Studio LoRA Train
      if (req.method === 'POST' && url.pathname === '/api/studio/lora/train') {
        const defaultPairs: ContrastivePair[] = [
          {
            anchor: 'EADDRINUSE port already bound',
            positive: 'Route IPC over singleton \\\\.\\pipe\\b8g-host-bus',
            negative: 'CSS table border alignment',
          },
          {
            anchor: 'untagged HTML template B8G022',
            positive: 'Prefix with html`<!DOCTYPE html>` tagged template',
            negative: 'GPU shader fog attenuation',
          },
        ];
        const pairs = Array.isArray(body.pairs) && body.pairs.length > 0
          ? (body.pairs as ContrastivePair[])
          : defaultPairs;
        const result = await studio.trainLora({
          taskId: (String(body.taskId ?? 'retrieval.query')) as TaskAdapterId,
          strategy: (body.strategy === 'merge-then-stack' ? 'merge-then-stack' : 'direct-task-finetune') as LoraStrategy,
          pairs,
          steps: Number(body.steps ?? 16),
          lr: Number(body.lr ?? 0.08),
          rank: Number(body.rank ?? 8),
          quant: (body.quant === 'fp16' || body.quant === 'q8' ? body.quant : 'nf4') as QuantizationKind,
        });
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ ok: true, ...result }));
        return;
      }

      // Studio LoRA Merge W' = W + (alpha/r) * B * A -> CAS
      if (req.method === 'POST' && url.pathname === '/api/studio/lora/merge') {
        const taskId = String(body.taskId ?? 'retrieval.query') as TaskAdapterId;
        const merged = studio.mergeAndPersistAdapter(taskId);
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ ok: true, ...merged }));
        return;
      }

      // Ollama Modelfile Parser
      if (req.method === 'POST' && url.pathname === '/api/studio/modelfile') {
        const parsed = parseOllamaModelfile(String(body.modelfile ?? ''));
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ ok: true, parsed }));
        return;
      }

      // Unsloth Dataset Validation & Export Plan
      if (req.method === 'POST' && url.pathname === '/api/studio/unsloth/plan') {
        const format = String(body.datasetFormat ?? 'contrastive-triplet') as DatasetFormat;
        const datasetValidation = Array.isArray(body.rows) ? validateDataset(format, body.rows) : undefined;
        const plan = buildUnslothExportPlan({
          recipeId: String(body.recipeId ?? 'qlora-nf4') as UnslothExportConfig['recipeId'],
          baseModel: String(body.baseModel ?? 'unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit'),
          adapterName: String(body.adapterName ?? 'awesome-os-domain-lora'),
          paramBillions: Number(body.paramBillions ?? 7),
          rank: Number(body.rank ?? 16),
          quant: (String(body.quant ?? 'Q4_K_M')) as UnslothExportConfig['quant'],
        });
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ ok: true, datasetValidation, plan }));
        return;
      }

      // Google Colab GPU Host Node + .ipynb Generator
      if (req.method === 'POST' && url.pathname === '/api/studio/colab/spawn') {
        const spawned = studio.spawnColabGpuIntersection({
          tenantId: String(body.tenantId ?? 'root-host'),
          label: String(body.label ?? 'Colab-L4-Node'),
          gpuTier: (String(body.gpuTier ?? 'L4-24GB')) as ColabGpuTier,
          notebookOpts: {
            recipe: String(body.recipe ?? 'qlora-nf4') as GenerateColabNotebookOptions['recipe'],
            baseModel: String(body.baseModel ?? 'unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit'),
            rank: Number(body.rank ?? 16),
            quant: String(body.quant ?? 'Q4_K_M'),
          },
        });
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ ok: true, ...spawned }));
        return;
      }

      res.writeHead(404, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: 'Not found', pathname: url.pathname }));
    } catch (err) {
      res.writeHead(500, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
    }
  });

  await new Promise<void>((resolveReady, rejectReady) => {
    server.once('error', rejectReady);
    server.listen(port, host, () => resolveReady());
  });

  const addr = server.address();
  const actualPort = typeof addr === 'object' && addr ? addr.port : port;
  return {
    url: `http://${host}:${actualPort}`,
    studio,
    close: () => new Promise<void>((r) => server.close(() => r())),
  };
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  const workspaceRoot = resolve(fileURLToPath(import.meta.url), '..', '..', '..');
  const srv = await startLlmStudioServer({
    port: 7497,
    casModelsDir: join(workspaceRoot, '.b8g-cas', 'models'),
  });
  console.log(`[b8g-llm-studio] Sovereign LLM, Embeddings & Unsloth Studio listening at ${srv.url}`);
}
