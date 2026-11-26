import assert from 'node:assert/strict';
import * as fsPromises from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { describe, it } from 'node:test';

import {
  GGUF_MAGIC,
  UnifiedModelHub,
  buildLlamaCppLaunchArgs,
  buildMinimalGgufHeader,
  detectQuantizationTag,
  parseGgufHeader,
  parseOllamaModelfile,
  searchHuggingFaceGguf,
} from './hub-gguf.mts';

describe('GGUF v2/v3 Binary Header Parser', () => {
  it('parses valid GGUF v2 and v3 headers built via buildMinimalGgufHeader', () => {
    const v3 = buildMinimalGgufHeader({ version: 3, tensorCount: 291, kvCount: 28 });
    const parsedV3 = parseGgufHeader(v3, 'qwen2.5-coder-7b-instruct-q4_k_m.gguf');
    assert.equal(parsedV3.magic, GGUF_MAGIC);
    assert.equal(parsedV3.magicText, 'GGUF');
    assert.equal(parsedV3.version, 3);
    assert.equal(parsedV3.tensorCount, 291);
    assert.equal(parsedV3.kvCount, 28);
    assert.equal(parsedV3.quantization, 'Q4_K_M');

    const v2 = buildMinimalGgufHeader({ version: 2, tensorCount: 128n, kvCount: 16n });
    const parsedV2 = parseGgufHeader(v2, 'jina-embeddings-v3-Q8_0.gguf');
    assert.equal(parsedV2.version, 2);
    assert.equal(parsedV2.tensorCount, 128);
    assert.equal(parsedV2.kvCount, 16);
    assert.equal(parsedV2.quantization, 'Q8_0');
  });

  it('rejects invalid magic, truncated buffers, and unsupported versions', () => {
    assert.throws(() => parseGgufHeader(new Uint8Array(12)), /too short/i);

    const badMagic = buildMinimalGgufHeader({ version: 3 });
    badMagic[0] = 0x00;
    assert.throws(() => parseGgufHeader(badMagic), /Invalid GGUF magic/i);

    const badVersion = buildMinimalGgufHeader({ version: 9 });
    assert.throws(() => parseGgufHeader(badVersion), /Unsupported GGUF version/i);
  });

  it('detects quantization tags from filenames accurately', () => {
    assert.equal(detectQuantizationTag('model-Q4_K_M.gguf'), 'Q4_K_M');
    assert.equal(detectQuantizationTag('model.q5_k_m.gguf'), 'Q5_K_M');
    assert.equal(detectQuantizationTag('embed-Q8_0.gguf'), 'Q8_0');
    assert.equal(detectQuantizationTag('weights-F16.gguf'), 'F16');
    assert.equal(detectQuantizationTag('router-nf4.gguf'), 'NF4');
    assert.equal(detectQuantizationTag('plain-model.gguf'), 'UNKNOWN');
  });
});

describe('Ollama Modelfile Parser', () => {
  it('parses FROM, PARAMETER, SYSTEM, ADAPTER, TEMPLATE, and ignores comments', () => {
    const modelfile = `
# Sovereign coder profile
FROM ./models/qwen2.5-coder-7b-q4_k_m.gguf

PARAMETER temperature 0.2
PARAMETER num_ctx 32768
PARAMETER mirostat false
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

ADAPTER ./loras/b8g-ts-adapter.gguf
ADAPTER ./loras/b8g-error-repair.gguf

SYSTEM """
You are a sovereign B8G TypeScript assistant.
Always emit strict erasableSyntaxOnly code.
"""

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}"""
`;

    const spec = parseOllamaModelfile(modelfile);
    assert.equal(spec.from, './models/qwen2.5-coder-7b-q4_k_m.gguf');
    assert.equal(spec.parameters.temperature, 0.2);
    assert.equal(spec.parameters.num_ctx, 32768);
    assert.equal(spec.parameters.mirostat, false);
    assert.deepEqual(spec.parameters.stop, ['<|im_end|>', '<|endoftext|>']);
    assert.equal(spec.adapter, './loras/b8g-ts-adapter.gguf');
    assert.deepEqual(spec.adapters, [
      './loras/b8g-ts-adapter.gguf',
      './loras/b8g-error-repair.gguf',
    ]);
    assert.ok(spec.system?.includes('sovereign B8G TypeScript assistant'));
    assert.ok(spec.template?.includes('{{ .System }}'));
  });
});

describe('UnifiedModelHub Catalog & Local GGUF Scanner', () => {
  it('seeds all 5 sovereign/recommended built-in entries including Jina v3 5 task LoRAs', () => {
    const hub = new UnifiedModelHub();
    const models = hub.listModels();
    const ids = models.map((m) => m.id);

    assert.ok(ids.includes('lumiBERT-local'));
    assert.ok(ids.includes('jina-embeddings-v3-gguf'));
    assert.ok(ids.includes('qwen2.5-coder-7b-q4_k_m'));
    assert.ok(ids.includes('llama-3.2-3b-instruct-q4_k_m'));
    assert.ok(ids.includes('b8g-moe-router'));

    const jina = hub.getModel('jina-embeddings-v3-gguf');
    assert.equal(jina?.kind, 'embed');
    assert.equal(jina?.taskAdapters?.length, 5);

    const openai = hub.toOpenAiModelsResponse();
    assert.equal(openai.object, 'list');
    assert.equal(openai.data.length, 5);

    const ollama = hub.toOllamaTagsResponse();
    assert.equal(ollama.models.length, 5);
  });

  it('scans a local directory for real binary .gguf files and skips corrupt files safely', async () => {
    const tmpDir = await fsPromises.mkdtemp(path.join(os.tmpdir(), 'b8g-hub-gguf-'));
    try {
      const validFile = path.join(tmpDir, 'mistral-7b-Q5_K_M.gguf');
      const header = buildMinimalGgufHeader({ version: 3, tensorCount: 201, kvCount: 19 });
      await fsPromises.writeFile(validFile, header);

      const corruptFile = path.join(tmpDir, 'corrupt-Q4_K_M.gguf');
      await fsPromises.writeFile(corruptFile, new Uint8Array([1, 2, 3, 4]));

      const nonGguf = path.join(tmpDir, 'README.md');
      await fsPromises.writeFile(nonGguf, '# ignored');

      const hub = new UnifiedModelHub({ baseDir: tmpDir, scanDirs: [tmpDir] });
      const scan = await hub.scanLocalDirectories([tmpDir]);

      assert.equal(scan.discovered.length, 1);
      assert.equal(scan.discovered[0]?.id, 'mistral-7b-Q5_K_M');
      assert.equal(scan.discovered[0]?.quantization, 'Q5_K_M');
      assert.equal(scan.discovered[0]?.ggufHeader?.version, 3);
      assert.equal(scan.discovered[0]?.ggufHeader?.tensorCount, 201);
      assert.equal(scan.skipped.length, 1);
      assert.ok(scan.skipped[0]?.filePath.endsWith('corrupt-Q4_K_M.gguf'));

      const localOnly = hub.listModels({ source: 'local' });
      assert.equal(localOnly.length, 1);
      assert.equal(localOnly[0]?.id, 'mistral-7b-Q5_K_M');
    } finally {
      await fsPromises.rm(tmpDir, { recursive: true, force: true });
    }
  });
});

describe('Hugging Face GGUF Search (Online & Offline Fallback)', () => {
  it('parses live Hugging Face API JSON responses cleanly', async () => {
    let capturedUrl = '';
    const mockFetch: typeof fetch = async (input) => {
      capturedUrl = String(input);
      return new Response(
        JSON.stringify([
          {
            id: 'bartowski/Qwen2.5-Coder-7B-Instruct-GGUF',
            downloads: 98765,
            likes: 432,
            tags: ['gguf', 'qwen2'],
            lastModified: '2026-11-01T00:00:00.000Z',
          },
        ]),
        { status: 200, headers: { 'content-type': 'application/json' } },
      );
    };

    const res = await searchHuggingFaceGguf('qwen2.5 coder', 5, { fetchFn: mockFetch });
    assert.ok(capturedUrl.includes('huggingface.co/api/models'));
    assert.ok(capturedUrl.includes('filter=gguf'));
    assert.ok(capturedUrl.includes('limit=5'));
    assert.equal(res.offlineFallback, false);
    assert.equal(res.source, 'huggingface');
    assert.equal(res.entries.length, 1);
    assert.equal(res.entries[0]?.repoId, 'bartowski/Qwen2.5-Coder-7B-Instruct-GGUF');
    assert.equal(res.entries[0]?.downloads, 98765);
  });

  it('gracefully falls back to filtered local/curated catalog when offline or failing', async () => {
    const failingFetch: typeof fetch = async () => {
      throw new Error('ENOTFOUND huggingface.co');
    };

    const hub = new UnifiedModelHub();
    const res = await hub.searchHuggingFace('jina', 5, { fetchFn: failingFetch });
    assert.equal(res.offlineFallback, true);
    assert.equal(res.source, 'offline-fallback');
    assert.ok(res.error?.includes('ENOTFOUND'));
    assert.equal(res.entries.length, 1);
    assert.equal(res.entries[0]?.id, 'jina-embeddings-v3-gguf');
  });
});

describe('llama.cpp Launch Args Builder', () => {
  it('generates exact llama-server args including flash-attn, embedding, and lora', () => {
    const args = buildLlamaCppLaunchArgs({
      modelPath: '.b8g-cas/models/jina-embeddings-v3-Q8_0.gguf',
      host: '127.0.0.1',
      port: 8091,
      contextSize: 8192,
      gpuLayers: 99,
      flashAttn: true,
      embedding: true,
      lora: '.b8g-cas/loras/retrieval-query.gguf',
    });

    assert.deepEqual(args, [
      '-m',
      '.b8g-cas/models/jina-embeddings-v3-Q8_0.gguf',
      '--host',
      '127.0.0.1',
      '--port',
      '8091',
      '-c',
      '8192',
      '-ngl',
      '99',
      '--flash-attn',
      '--embedding',
      '--lora',
      '.b8g-cas/loras/retrieval-query.gguf',
    ]);
  });

  it('validates required modelPath and port range', () => {
    assert.throws(() => buildLlamaCppLaunchArgs({ modelPath: '' }), /non-empty modelPath/i);
    assert.throws(
      () => buildLlamaCppLaunchArgs({ modelPath: 'm.gguf', port: 70000 }),
      /Invalid port/i,
    );
  });
});
