import * as fs from 'node:fs';
import * as fsPromises from 'node:fs/promises';
import * as path from 'node:path';

/**
 * GGUF magic bytes in little-endian: 'G' (0x47), 'G' (0x47), 'U' (0x55), 'F' (0x46) -> 0x46554747.
 */
export const GGUF_MAGIC = 0x46554747;
export const GGUF_HEADER_MIN_BYTES = 24;
export const DEFAULT_HF_TIMEOUT_MS = 4000;
export const DEFAULT_LOCAL_MODEL_DIR = '.b8g-cas/models';

export type ModelKind = 'chat' | 'embed' | 'routing' | 'multimodal';
export type ModelSource = 'builtin' | 'local' | 'huggingface' | 'ollama';

export interface GgufHeaderInfo {
  magic: number;
  magicText: 'GGUF';
  version: number;
  tensorCount: number;
  kvCount: number;
  quantization: string;
}

export interface BuildGgufHeaderOptions {
  version?: number;
  tensorCount?: number | bigint;
  kvCount?: number | bigint;
}

export interface OllamaModelfileMessage {
  role: string;
  content: string;
}

export interface OllamaModelfileSpec {
  from: string;
  parameters: Record<string, string | number | boolean | string[]>;
  system?: string;
  adapter?: string;
  adapters: string[];
  template?: string;
  license?: string;
  messages: OllamaModelfileMessage[];
}

export interface ModelRegistryEntry {
  id: string;
  name: string;
  source: ModelSource;
  kind: ModelKind;
  quantization: string;
  parameterSize?: string;
  sizeBytes?: number;
  contextLength?: number;
  embeddingDim?: number;
  localPath?: string;
  repoId?: string;
  downloads?: number;
  likes?: number;
  tags: string[];
  taskAdapters?: string[];
  ggufHeader?: GgufHeaderInfo;
  recommended?: boolean;
  loaded?: boolean;
  description?: string;
  modifiedAt?: string;
}

export interface HfSearchOptions {
  timeoutMs?: number;
  fetchFn?: typeof fetch;
  fallbackCatalog?: readonly ModelRegistryEntry[];
  forceOffline?: boolean;
}

export interface HfSearchResult {
  query: string;
  limit: number;
  source: 'huggingface' | 'offline-fallback';
  offlineFallback: boolean;
  entries: ModelRegistryEntry[];
  error?: string;
}

export interface LlamaCppLaunchOptions {
  modelPath: string;
  host?: string;
  port?: number;
  contextSize?: number;
  gpuLayers?: number;
  flashAttn?: boolean;
  embedding?: boolean;
  lora?: string;
  loras?: readonly string[];
  threads?: number;
  batchSize?: number;
  parallel?: number;
  alias?: string;
  extraArgs?: readonly string[];
}

export interface LocalScanError {
  filePath: string;
  reason: string;
}

export interface LocalScanResult {
  scannedDirs: string[];
  discovered: ModelRegistryEntry[];
  skipped: LocalScanError[];
}

export interface UnifiedModelHubOptions {
  baseDir?: string;
  scanDirs?: readonly string[];
  seedDefaults?: boolean;
}

export interface ModelFilter {
  kind?: ModelKind;
  source?: ModelSource;
  query?: string;
}

const KNOWN_QUANTIZATION_PATTERNS: ReadonlyArray<{ pattern: RegExp; tag: string }> = [
  { pattern: /(?:^|[^a-z0-9])(iq4_xs)(?:[^a-z0-9]|$)/i, tag: 'IQ4_XS' },
  { pattern: /(?:^|[^a-z0-9])(iq4_nl)(?:[^a-z0-9]|$)/i, tag: 'IQ4_NL' },
  { pattern: /(?:^|[^a-z0-9])(iq3_xxs)(?:[^a-z0-9]|$)/i, tag: 'IQ3_XXS' },
  { pattern: /(?:^|[^a-z0-9])(iq3_xs)(?:[^a-z0-9]|$)/i, tag: 'IQ3_XS' },
  { pattern: /(?:^|[^a-z0-9])(iq2_xxs)(?:[^a-z0-9]|$)/i, tag: 'IQ2_XXS' },
  { pattern: /(?:^|[^a-z0-9])(iq2_xs)(?:[^a-z0-9]|$)/i, tag: 'IQ2_XS' },
  { pattern: /(?:^|[^a-z0-9])(q8_0)(?:[^a-z0-9]|$)/i, tag: 'Q8_0' },
  { pattern: /(?:^|[^a-z0-9])(q6_k)(?:[^a-z0-9]|$)/i, tag: 'Q6_K' },
  { pattern: /(?:^|[^a-z0-9])(q5_k_m)(?:[^a-z0-9]|$)/i, tag: 'Q5_K_M' },
  { pattern: /(?:^|[^a-z0-9])(q5_k_s)(?:[^a-z0-9]|$)/i, tag: 'Q5_K_S' },
  { pattern: /(?:^|[^a-z0-9])(q5_0)(?:[^a-z0-9]|$)/i, tag: 'Q5_0' },
  { pattern: /(?:^|[^a-z0-9])(q5_1)(?:[^a-z0-9]|$)/i, tag: 'Q5_1' },
  { pattern: /(?:^|[^a-z0-9])(q4_k_m)(?:[^a-z0-9]|$)/i, tag: 'Q4_K_M' },
  { pattern: /(?:^|[^a-z0-9])(q4_k_s)(?:[^a-z0-9]|$)/i, tag: 'Q4_K_S' },
  { pattern: /(?:^|[^a-z0-9])(q4_0)(?:[^a-z0-9]|$)/i, tag: 'Q4_0' },
  { pattern: /(?:^|[^a-z0-9])(q4_1)(?:[^a-z0-9]|$)/i, tag: 'Q4_1' },
  { pattern: /(?:^|[^a-z0-9])(q3_k_l)(?:[^a-z0-9]|$)/i, tag: 'Q3_K_L' },
  { pattern: /(?:^|[^a-z0-9])(q3_k_m)(?:[^a-z0-9]|$)/i, tag: 'Q3_K_M' },
  { pattern: /(?:^|[^a-z0-9])(q3_k_s)(?:[^a-z0-9]|$)/i, tag: 'Q3_K_S' },
  { pattern: /(?:^|[^a-z0-9])(q2_k)(?:[^a-z0-9]|$)/i, tag: 'Q2_K' },
  { pattern: /(?:^|[^a-z0-9])(bf16)(?:[^a-z0-9]|$)/i, tag: 'BF16' },
  { pattern: /(?:^|[^a-z0-9])(fp16|f16)(?:[^a-z0-9]|$)/i, tag: 'F16' },
  { pattern: /(?:^|[^a-z0-9])(fp32|f32)(?:[^a-z0-9]|$)/i, tag: 'F32' },
  { pattern: /(?:^|[^a-z0-9])(nf4)(?:[^a-z0-9]|$)/i, tag: 'NF4' },
];

/**
 * Detects a GGUF/LoRA quantization tag (e.g., Q4_K_M, Q5_K_M, Q8_0, F16, NF4) from a filename or model id.
 */
export function detectQuantizationTag(filename?: string): string {
  if (!filename) return 'UNKNOWN';
  const normalized = path.basename(filename);
  for (const entry of KNOWN_QUANTIZATION_PATTERNS) {
    if (entry.pattern.test(normalized)) {
      return entry.tag;
    }
  }
  return 'UNKNOWN';
}

/**
 * Builds a minimal valid 24-byte GGUF v2/v3 binary header (useful for tests and synthetic fixtures).
 */
export function buildMinimalGgufHeader(options: BuildGgufHeaderOptions = {}): Uint8Array {
  const version = options.version ?? 3;
  if (!Number.isInteger(version) || version < 1) {
    throw new RangeError(`Invalid GGUF version: ${String(version)}`);
  }

  const tensorCount = BigInt(options.tensorCount ?? 0);
  const kvCount = BigInt(options.kvCount ?? 0);
  if (tensorCount < 0n || kvCount < 0n) {
    throw new RangeError('GGUF tensorCount and kvCount must be non-negative');
  }

  const bytes = new Uint8Array(GGUF_HEADER_MIN_BYTES);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  view.setUint32(0, GGUF_MAGIC, true);
  view.setUint32(4, version, true);
  view.setBigUint64(8, tensorCount, true);
  view.setBigUint64(16, kvCount, true);
  return bytes;
}

/**
 * Parses a binary GGUF v2/v3 header from a Uint8Array.
 * Validates the 4-byte magic (0x46554747 = 'GGUF'), u32 LE version, u64 LE tensorCount, and u64 LE kvCount.
 */
export function parseGgufHeader(bytes: Uint8Array, filename?: string): GgufHeaderInfo {
  if (!(bytes instanceof Uint8Array)) {
    throw new TypeError('parseGgufHeader expects a Uint8Array');
  }
  if (bytes.byteLength < GGUF_HEADER_MIN_BYTES) {
    throw new RangeError(
      `GGUF header too short: expected at least ${GGUF_HEADER_MIN_BYTES} bytes, got ${bytes.byteLength}`,
    );
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = view.getUint32(0, true);
  if (magic !== GGUF_MAGIC) {
    throw new Error(
      `Invalid GGUF magic: expected 0x${GGUF_MAGIC.toString(16)} ('GGUF'), got 0x${magic.toString(16).padStart(8, '0')}`,
    );
  }

  const version = view.getUint32(4, true);
  if (version !== 2 && version !== 3) {
    throw new Error(`Unsupported GGUF version: ${version} (supported: v2, v3)`);
  }

  const tensorCountBig = view.getBigUint64(8, true);
  const kvCountBig = view.getBigUint64(16, true);
  const maxSafe = BigInt(Number.MAX_SAFE_INTEGER);
  if (tensorCountBig > maxSafe || kvCountBig > maxSafe) {
    throw new RangeError('GGUF tensorCount or kvCount exceeds Number.MAX_SAFE_INTEGER');
  }

  return {
    magic,
    magicText: 'GGUF',
    version,
    tensorCount: Number(tensorCountBig),
    kvCount: Number(kvCountBig),
    quantization: detectQuantizationTag(filename),
  };
}

function parseParameterValue(raw: string): string | number | boolean {
  const trimmed = raw.trim();
  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"') && trimmed.length >= 2) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'") && trimmed.length >= 2)
  ) {
    return trimmed.slice(1, -1);
  }
  if (trimmed === 'true') return true;
  if (trimmed === 'false') return false;
  if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(trimmed)) {
    const num = Number(trimmed);
    if (Number.isFinite(num)) return num;
  }
  return trimmed;
}

function unquoteSingleLine(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.startsWith('"""') && trimmed.endsWith('"""') && trimmed.length >= 6) {
    return trimmed.slice(3, -3);
  }
  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"') && trimmed.length >= 2) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'") && trimmed.length >= 2)
  ) {
    return trimmed
      .slice(1, -1)
      .replace(/\\n/g, '\n')
      .replace(/\\t/g, '\t')
      .replace(/\\"/g, '"')
      .replace(/\\\\/g, '\\');
  }
  return trimmed;
}

/**
 * Parses an Ollama Modelfile into a structured OllamaModelfileSpec.
 * Supports FROM, PARAMETER <k> <v>, SYSTEM "..." / """...""", ADAPTER <path>,
 * TEMPLATE "..." / """...""", LICENSE, and MESSAGE <role> <content>.
 */
export function parseOllamaModelfile(text: string): OllamaModelfileSpec {
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const spec: OllamaModelfileSpec = {
    from: '',
    parameters: {},
    adapters: [],
    messages: [],
  };

  let i = 0;
  while (i < lines.length) {
    const rawLine = lines[i] ?? '';
    const trimmed = rawLine.trim();
    i += 1;

    if (!trimmed || trimmed.startsWith('#')) {
      continue;
    }

    const match = /^([A-Za-z]+)\s+([\s\S]*)$/.exec(trimmed);
    if (!match) {
      continue;
    }

    const directive = match[1]!.toUpperCase();
    let rest = match[2]!.trim();

    const readBlockValue = (initial: string): string => {
      if (initial.startsWith('"""')) {
        const afterOpen = initial.slice(3);
        const closeInline = afterOpen.indexOf('"""');
        if (closeInline !== -1) {
          return afterOpen.slice(0, closeInline);
        }
        const collected: string[] = [afterOpen];
        while (i < lines.length) {
          const nextLine = lines[i] ?? '';
          i += 1;
          const closeIdx = nextLine.indexOf('"""');
          if (closeIdx !== -1) {
            collected.push(nextLine.slice(0, closeIdx));
            break;
          }
          collected.push(nextLine);
        }
        return collected.join('\n').replace(/^\n/, '').replace(/\n$/, '');
      }
      return unquoteSingleLine(initial);
    };

    switch (directive) {
      case 'FROM': {
        spec.from = unquoteSingleLine(rest);
        break;
      }
      case 'PARAMETER': {
        const paramMatch = /^([^\s]+)\s+([\s\S]+)$/.exec(rest);
        if (paramMatch) {
          const key = paramMatch[1]!.trim();
          const val = parseParameterValue(paramMatch[2]!);
          const existing = spec.parameters[key];
          if (existing === undefined) {
            spec.parameters[key] = val;
          } else if (Array.isArray(existing)) {
            existing.push(String(val));
          } else {
            spec.parameters[key] = [String(existing), String(val)];
          }
        }
        break;
      }
      case 'SYSTEM': {
        spec.system = readBlockValue(rest);
        break;
      }
      case 'ADAPTER': {
        const adapterPath = unquoteSingleLine(rest);
        if (adapterPath) {
          spec.adapters.push(adapterPath);
          if (!spec.adapter) {
            spec.adapter = adapterPath;
          }
        }
        break;
      }
      case 'TEMPLATE': {
        spec.template = readBlockValue(rest);
        break;
      }
      case 'LICENSE': {
        spec.license = readBlockValue(rest);
        break;
      }
      case 'MESSAGE': {
        const msgMatch = /^([^\s]+)\s+([\s\S]+)$/.exec(rest);
        if (msgMatch) {
          spec.messages.push({
            role: msgMatch[1]!.trim(),
            content: readBlockValue(msgMatch[2]!.trim()),
          });
        }
        break;
      }
      default:
        break;
    }
  }

  return spec;
}

/**
 * Built-in sovereign / recommended model catalog entries.
 */
export function createBuiltInCatalog(): ModelRegistryEntry[] {
  return [
    {
      id: 'lumiBERT-local',
      name: 'lumiBERT Local (2.8ms Sovereign Embedder)',
      source: 'builtin',
      kind: 'embed',
      quantization: 'F16',
      parameterSize: '110M',
      sizeBytes: 220_000_000,
      contextLength: 512,
      embeddingDim: 768,
      tags: ['embed', 'sovereign', 'error-repair', 'gguf', 'local'],
      recommended: true,
      loaded: true,
      description: 'Ultra-low-latency local embedding & error auto-repair encoder.',
    },
    {
      id: 'jina-embeddings-v3-gguf',
      name: 'Jina Embeddings v3 GGUF (5 Task LoRAs + Matryoshka)',
      source: 'builtin',
      kind: 'embed',
      quantization: 'Q8_0',
      parameterSize: '570M',
      sizeBytes: 615_000_000,
      contextLength: 8192,
      embeddingDim: 1024,
      repoId: 'jinaai/jina-embeddings-v3',
      tags: ['embed', 'matryoshka', 'lora', 'gguf', 'multilingual'],
      taskAdapters: [
        'retrieval.query',
        'retrieval.passage',
        'text-matching',
        'classification',
        'separation',
      ],
      recommended: true,
      loaded: false,
      description: 'Multilingual 8K-context embedding model with 5 task-specific LoRA adapters.',
    },
    {
      id: 'qwen2.5-coder-7b-q4_k_m',
      name: 'Qwen 2.5 Coder 7B Instruct (Q4_K_M)',
      source: 'builtin',
      kind: 'chat',
      quantization: 'Q4_K_M',
      parameterSize: '7B',
      sizeBytes: 4_683_000_000,
      contextLength: 32768,
      repoId: 'Qwen/Qwen2.5-Coder-7B-Instruct-GGUF',
      tags: ['chat', 'coder', 'gguf', 'q4_k_m', 'instruct'],
      recommended: true,
      loaded: false,
      description: 'High-accuracy local coding assistant quantized to Q4_K_M GGUF.',
    },
    {
      id: 'llama-3.2-3b-instruct-q4_k_m',
      name: 'Llama 3.2 3B Instruct (Q4_K_M)',
      source: 'builtin',
      kind: 'chat',
      quantization: 'Q4_K_M',
      parameterSize: '3B',
      sizeBytes: 2_019_000_000,
      contextLength: 131072,
      repoId: 'bartowski/Llama-3.2-3B-Instruct-GGUF',
      tags: ['chat', 'llama', 'gguf', 'q4_k_m', 'instruct'],
      recommended: true,
      loaded: false,
      description: 'Fast, lightweight 3B instruct model for sovereign edge chat.',
    },
    {
      id: 'b8g-moe-router',
      name: 'B8G MoE Intent & Expert Router',
      source: 'builtin',
      kind: 'routing',
      quantization: 'NF4',
      parameterSize: '180M',
      sizeBytes: 112_000_000,
      contextLength: 2048,
      tags: ['routing', 'moe', 'sovereign', 'nf4', 'local'],
      recommended: true,
      loaded: true,
      description: 'Sovereign Mixture-of-Experts router dispatching prompts to local specialists.',
    },
  ];
}

function cloneEntry(entry: ModelRegistryEntry): ModelRegistryEntry {
  return {
    ...entry,
    tags: [...entry.tags],
    taskAdapters: entry.taskAdapters ? [...entry.taskAdapters] : undefined,
    ggufHeader: entry.ggufHeader ? { ...entry.ggufHeader } : undefined,
  };
}

function inferModelKindFromName(name: string, tags: readonly string[] = []): ModelKind {
  const lower = `${name} ${tags.join(' ')}`.toLowerCase();
  if (lower.includes('router') || lower.includes('moe-router')) return 'routing';
  if (lower.includes('embed') || lower.includes('bert') || lower.includes('bge') || lower.includes('nomic')) {
    return 'embed';
  }
  if (lower.includes('vision') || lower.includes('vl') || lower.includes('llava')) {
    return 'multimodal';
  }
  return 'chat';
}

function filterEntriesByQuery(
  entries: readonly ModelRegistryEntry[],
  query: string,
  limit: number,
): ModelRegistryEntry[] {
  const q = query.trim().toLowerCase();
  const cappedLimit = Math.max(1, Math.min(100, Math.floor(limit)));
  if (!q) {
    return entries.slice(0, cappedLimit).map(cloneEntry);
  }
  const tokens = q.split(/\s+/).filter(Boolean);
  const matched = entries.filter((entry) => {
    const haystack = [
      entry.id,
      entry.name,
      entry.repoId ?? '',
      entry.kind,
      entry.quantization,
      entry.description ?? '',
      ...entry.tags,
    ]
      .join(' ')
      .toLowerCase();
    return tokens.every((tok) => haystack.includes(tok));
  });
  return matched.slice(0, cappedLimit).map(cloneEntry);
}

interface HuggingFaceApiModel {
  id?: string;
  modelId?: string;
  downloads?: number;
  likes?: number;
  tags?: string[];
  lastModified?: string;
}

/**
 * Searches Hugging Face Hub for GGUF models with a 4-second timeout and graceful offline fallback.
 * Never throws on offline/timeout/HTTP errors; falls back to filtered local/curated entries.
 */
export async function searchHuggingFaceGguf(
  query: string,
  limit = 10,
  options: HfSearchOptions = {},
): Promise<HfSearchResult> {
  const safeLimit = Math.max(1, Math.min(100, Math.floor(limit)));
  const timeoutMs = options.timeoutMs ?? DEFAULT_HF_TIMEOUT_MS;
  const fallbackPool = options.fallbackCatalog ?? createBuiltInCatalog();

  const buildFallback = (reason: string): HfSearchResult => {
    const filtered = filterEntriesByQuery(fallbackPool, query, safeLimit);
    const entries =
      filtered.length > 0 ? filtered : fallbackPool.slice(0, safeLimit).map(cloneEntry);
    return {
      query,
      limit: safeLimit,
      source: 'offline-fallback',
      offlineFallback: true,
      entries,
      error: reason,
    };
  };

  if (options.forceOffline) {
    return buildFallback('Offline mode enabled');
  }

  const fetchImpl = options.fetchFn ?? globalThis.fetch;
  if (typeof fetchImpl !== 'function') {
    return buildFallback('Fetch API unavailable');
  }

  const url =
    `https://huggingface.co/api/models?search=${encodeURIComponent(query)}` +
    `&filter=gguf&sort=downloads&direction=-1&limit=${safeLimit}`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetchImpl(url, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });

    if (!response.ok) {
      return buildFallback(`Hugging Face HTTP ${response.status}`);
    }

    const payload = (await response.json()) as unknown;
    if (!Array.isArray(payload)) {
      return buildFallback('Unexpected Hugging Face API payload');
    }

    const entries: ModelRegistryEntry[] = [];
    for (const raw of payload as HuggingFaceApiModel[]) {
      const repoId = typeof raw.id === 'string' ? raw.id : typeof raw.modelId === 'string' ? raw.modelId : '';
      if (!repoId) continue;
      const tags = Array.isArray(raw.tags) ? raw.tags.filter((t): t is string => typeof t === 'string') : ['gguf'];
      const quant = detectQuantizationTag(repoId);
      entries.push({
        id: repoId,
        name: repoId.split('/').pop() ?? repoId,
        source: 'huggingface',
        kind: inferModelKindFromName(repoId, tags),
        quantization: quant === 'UNKNOWN' ? 'GGUF' : quant,
        repoId,
        downloads: typeof raw.downloads === 'number' ? raw.downloads : 0,
        likes: typeof raw.likes === 'number' ? raw.likes : 0,
        tags,
        modifiedAt: typeof raw.lastModified === 'string' ? raw.lastModified : undefined,
      });
    }

    return {
      query,
      limit: safeLimit,
      source: 'huggingface',
      offlineFallback: false,
      entries,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return buildFallback(message);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Builds exact `llama-server` CLI arguments for `llama.cpp`.
 */
export function buildLlamaCppLaunchArgs(opts: LlamaCppLaunchOptions): string[] {
  if (!opts.modelPath || !opts.modelPath.trim()) {
    throw new Error('buildLlamaCppLaunchArgs requires a non-empty modelPath');
  }

  const args: string[] = ['-m', opts.modelPath.trim()];

  if (opts.host !== undefined && opts.host.trim() !== '') {
    args.push('--host', opts.host.trim());
  }

  if (opts.port !== undefined) {
    if (!Number.isInteger(opts.port) || opts.port < 1 || opts.port > 65535) {
      throw new RangeError(`Invalid port: ${String(opts.port)}`);
    }
    args.push('--port', String(opts.port));
  }

  if (opts.contextSize !== undefined) {
    if (!Number.isInteger(opts.contextSize) || opts.contextSize <= 0) {
      throw new RangeError(`Invalid contextSize: ${String(opts.contextSize)}`);
    }
    args.push('-c', String(opts.contextSize));
  }

  if (opts.gpuLayers !== undefined) {
    if (!Number.isInteger(opts.gpuLayers) || opts.gpuLayers < 0) {
      throw new RangeError(`Invalid gpuLayers: ${String(opts.gpuLayers)}`);
    }
    args.push('-ngl', String(opts.gpuLayers));
  }

  if (opts.flashAttn) {
    args.push('--flash-attn');
  }

  if (opts.embedding) {
    args.push('--embedding');
  }

  const loraList: string[] = [];
  if (opts.lora && opts.lora.trim()) {
    loraList.push(opts.lora.trim());
  }
  if (opts.loras) {
    for (const item of opts.loras) {
      if (item && item.trim()) {
        loraList.push(item.trim());
      }
    }
  }
  for (const loraPath of loraList) {
    args.push('--lora', loraPath);
  }

  if (opts.threads !== undefined && Number.isInteger(opts.threads) && opts.threads > 0) {
    args.push('-t', String(opts.threads));
  }

  if (opts.batchSize !== undefined && Number.isInteger(opts.batchSize) && opts.batchSize > 0) {
    args.push('-b', String(opts.batchSize));
  }

  if (opts.parallel !== undefined && Number.isInteger(opts.parallel) && opts.parallel > 0) {
    args.push('-np', String(opts.parallel));
  }

  if (opts.alias && opts.alias.trim()) {
    args.push('--alias', opts.alias.trim());
  }

  if (opts.extraArgs) {
    for (const extra of opts.extraArgs) {
      if (extra) args.push(extra);
    }
  }

  return args;
}

async function readGgufHeaderFromFile(filePath: string): Promise<GgufHeaderInfo> {
  const handle = await fsPromises.open(filePath, 'r');
  try {
    const buffer = new Uint8Array(GGUF_HEADER_MIN_BYTES);
    const { bytesRead } = await handle.read(buffer, 0, GGUF_HEADER_MIN_BYTES, 0);
    return parseGgufHeader(buffer.subarray(0, bytesRead), filePath);
  } finally {
    await handle.close();
  }
}

/**
 * Unified Hugging Face + Ollama + Local GGUF (`llama.cpp`) Model Registry & Parser.
 */
export class UnifiedModelHub {
  readonly baseDir: string;
  readonly scanDirs: string[];
  readonly #catalog = new Map<string, ModelRegistryEntry>();

  constructor(options: UnifiedModelHubOptions = {}) {
    this.baseDir = options.baseDir ?? process.cwd();
    const defaultDir = path.resolve(this.baseDir, DEFAULT_LOCAL_MODEL_DIR);
    const customDirs = (options.scanDirs ?? []).map((dir) => path.resolve(this.baseDir, dir));
    this.scanDirs = Array.from(new Set([defaultDir, ...customDirs]));

    if (options.seedDefaults !== false) {
      for (const entry of createBuiltInCatalog()) {
        this.#catalog.set(entry.id, cloneEntry(entry));
      }
    }
  }

  listModels(filter: ModelFilter = {}): ModelRegistryEntry[] {
    let items = Array.from(this.#catalog.values());
    if (filter.kind) {
      items = items.filter((m) => m.kind === filter.kind);
    }
    if (filter.source) {
      items = items.filter((m) => m.source === filter.source);
    }
    if (filter.query && filter.query.trim()) {
      return filterEntriesByQuery(items, filter.query, items.length || 1);
    }
    return items.map(cloneEntry);
  }

  getModel(id: string): ModelRegistryEntry | undefined {
    const found = this.#catalog.get(id);
    return found ? cloneEntry(found) : undefined;
  }

  registerModel(entry: ModelRegistryEntry): ModelRegistryEntry {
    const copy = cloneEntry(entry);
    this.#catalog.set(copy.id, copy);
    return cloneEntry(copy);
  }

  removeModel(id: string): boolean {
    return this.#catalog.delete(id);
  }

  /**
   * Scans configured directories (or custom override directories) for real `.gguf` files,
   * validates their binary headers, and registers them into the catalog.
   */
  async scanLocalDirectories(customDirs?: readonly string[]): Promise<LocalScanResult> {
    const dirsToScan = customDirs
      ? Array.from(new Set(customDirs.map((d) => path.resolve(this.baseDir, d))))
      : [...this.scanDirs];

    const discovered: ModelRegistryEntry[] = [];
    const skipped: LocalScanError[] = [];

    for (const dir of dirsToScan) {
      if (!fs.existsSync(dir)) {
        continue;
      }

      let entries: fs.Dirent[];
      try {
        entries = await fsPromises.readdir(dir, { withFileTypes: true });
      } catch (err) {
        skipped.push({
          filePath: dir,
          reason: err instanceof Error ? err.message : String(err),
        });
        continue;
      }

      for (const dirent of entries) {
        if (!dirent.isFile() || !dirent.name.toLowerCase().endsWith('.gguf')) {
          continue;
        }
        const fullPath = path.join(dir, dirent.name);
        try {
          const [header, stat] = await Promise.all([
            readGgufHeaderFromFile(fullPath),
            fsPromises.stat(fullPath),
          ]);
          const id = dirent.name.replace(/\.gguf$/i, '');
          const modelEntry: ModelRegistryEntry = {
            id,
            name: dirent.name,
            source: 'local',
            kind: inferModelKindFromName(dirent.name),
            quantization: header.quantization,
            sizeBytes: stat.size,
            localPath: fullPath,
            tags: ['local', 'gguf', `gguf-v${header.version}`, header.quantization.toLowerCase()],
            ggufHeader: header,
            modifiedAt: stat.mtime.toISOString(),
          };
          this.#catalog.set(modelEntry.id, cloneEntry(modelEntry));
          discovered.push(cloneEntry(modelEntry));
        } catch (err) {
          skipped.push({
            filePath: fullPath,
            reason: err instanceof Error ? err.message : String(err),
          });
        }
      }
    }

    return {
      scannedDirs: dirsToScan,
      discovered,
      skipped,
    };
  }

  async searchHuggingFace(
    query: string,
    limit = 10,
    options: Omit<HfSearchOptions, 'fallbackCatalog'> = {},
  ): Promise<HfSearchResult> {
    return searchHuggingFaceGguf(query, limit, {
      ...options,
      fallbackCatalog: this.listModels(),
    });
  }

  parseModelfile(text: string): OllamaModelfileSpec {
    return parseOllamaModelfile(text);
  }

  buildLaunchArgs(opts: LlamaCppLaunchOptions): string[] {
    return buildLlamaCppLaunchArgs(opts);
  }

  toOpenAiModelsResponse(): {
    object: 'list';
    data: Array<{
      id: string;
      object: 'model';
      created: number;
      owned_by: string;
      kind: ModelKind;
      quantization: string;
    }>;
  } {
    const nowSec = Math.floor(Date.now() / 1000);
    return {
      object: 'list',
      data: this.listModels().map((m) => ({
        id: m.id,
        object: 'model',
        created: m.modifiedAt ? Math.floor(Date.parse(m.modifiedAt) / 1000) || nowSec : nowSec,
        owned_by: m.source,
        kind: m.kind,
        quantization: m.quantization,
      })),
    };
  }

  toOllamaTagsResponse(): {
    models: Array<{
      name: string;
      model: string;
      size: number;
      modified_at: string;
      details: {
        format: string;
        quantization_level: string;
        parameter_size: string;
      };
    }>;
  } {
    const fallbackIso = new Date().toISOString();
    return {
      models: this.listModels().map((m) => ({
        name: m.id,
        model: m.id,
        size: m.sizeBytes ?? 0,
        modified_at: m.modifiedAt ?? fallbackIso,
        details: {
          format: 'gguf',
          quantization_level: m.quantization,
          parameter_size: m.parameterSize ?? 'unknown',
        },
      })),
    };
  }
}
