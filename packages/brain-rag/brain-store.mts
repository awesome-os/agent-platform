/**
 * B8G Universal Sovereign Brain — Multi-Tenant & Multi-Session Storage Engine.
 *
 * Isolates every tenant and session cleanly under:
 *   `<brainRoot>/tenants/<tenantId>/`
 *     - `tenant.json`        (metadata, workspace root, model config)
 *     - `index.json`         (SHA-256 incremental CAS code chunks + trigram vectors)
 *     - `memory.json`        (model-independent episodic & architectural memory)
 *     - `sessions/<sid>.json` (isolated per-session conversation history)
 */
import { createHash, randomBytes } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { extname, join, relative, resolve } from 'node:path';

export interface BrainTenant {
  tenantId: string;
  name: string;
  workspaceDir?: string;
  defaultModel: string;
  providerUrl: string;
  createdAt: string;
}

export interface BrainSession {
  sessionId: string;
  tenantId: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string; model?: string; ts: string }>;
}

export interface CodeChunk {
  id: string;
  file: string;
  startLine: number;
  endLine: number;
  fileHash: string;
  text: string;
  /** Compact 64-dim hashed char-trigram unit vector for zero-dep cosine retrieval. */
  vec: number[];
}

export interface MemoryItem {
  id: string;
  text: string;
  tags: string[];
  createdAt: string;
  vec: number[];
}

export interface SearchHit {
  source: 'code' | 'memory';
  score: number;
  file?: string;
  startLine?: number;
  endLine?: number;
  tags?: string[];
  text: string;
}

const INDEXABLE_EXTS = new Set([
  '.ts', '.mts', '.cts', '.tsx', '.js', '.mjs', '.jsx',
  '.c', '.cc', '.cpp', '.h', '.hpp', '.rs', '.zig', '.py', '.go',
  '.md', '.json', '.gn', '.gni', '.sh', '.ps1', '.html', '.css',
]);

const SKIP_DIRS = new Set([
  '.git', 'node_modules', 'out', 'dist', 'build', 'target', '.b8g-cas', 'third_party',
]);

/** Deterministic 64-dim normalized char-trigram embedding (zero external APIs required). */
export function embedTextLocal(text: string): number[] {
  const dims = 64;
  const vec = new Float64Array(dims);
  const norm = text.toLowerCase().replace(/\s+/g, ' ').trim();
  for (let i = 0; i + 3 <= norm.length; i++) {
    let h = 2166136261;
    h = Math.imul(h ^ norm.charCodeAt(i), 16777619);
    h = Math.imul(h ^ norm.charCodeAt(i + 1), 16777619);
    h = Math.imul(h ^ norm.charCodeAt(i + 2), 16777619);
    const idx = (h >>> 0) % dims;
    vec[idx] += 1;
  }
  let mag = 0;
  for (let i = 0; i < dims; i++) mag += vec[i] * vec[i];
  mag = Math.sqrt(mag) || 1;
  const out: number[] = new Array(dims);
  for (let i = 0; i < dims; i++) out[i] = Math.round((vec[i] / mag) * 10000) / 10000;
  return out;
}

function cosineSim(a: number[], b: number[]): number {
  let sum = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) sum += a[i] * b[i];
  return sum;
}

function lexicalScore(queryTerms: string[], text: string): number {
  const lower = text.toLowerCase();
  let hits = 0;
  for (const t of queryTerms) {
    if (lower.includes(t)) hits++;
  }
  return queryTerms.length > 0 ? hits / queryTerms.length : 0;
}

function sanitizeId(raw: string): string {
  const clean = raw.trim().toLowerCase().replace(/[^a-z0-9._-]/g, '-').replace(/^-+|-+$/g, '');
  if (!clean) throw new Error('Invalid identifier');
  return clean;
}

export class B8gBrainStore {
  readonly rootDir: string;

  constructor(rootDir = 'E:\\b8g-brains') {
    this.rootDir = resolve(rootDir);
    mkdirSync(join(this.rootDir, 'tenants'), { recursive: true });
  }

  private tenantDir(tenantId: string): string {
    return join(this.rootDir, 'tenants', sanitizeId(tenantId));
  }

  /** Generator: stamp out a brand-new isolated tenant brain. */
  createTenant(
    tenantIdRaw: string,
    opts: { name?: string; workspaceDir?: string; defaultModel?: string; providerUrl?: string } = {},
  ): BrainTenant {
    const tenantId = sanitizeId(tenantIdRaw);
    const dir = this.tenantDir(tenantId);
    mkdirSync(join(dir, 'sessions'), { recursive: true });

    const metaPath = join(dir, 'tenant.json');
    if (existsSync(metaPath)) {
      return JSON.parse(readFileSync(metaPath, 'utf8')) as BrainTenant;
    }

    const tenant: BrainTenant = {
      tenantId,
      name: opts.name || tenantId,
      workspaceDir: opts.workspaceDir ? resolve(opts.workspaceDir) : undefined,
      defaultModel: opts.defaultModel || 'openrouter/auto',
      providerUrl: opts.providerUrl || 'https://openrouter.ai/api/v1/chat/completions',
      createdAt: new Date().toISOString(),
    };
    writeFileSync(metaPath, JSON.stringify(tenant, null, 2), 'utf8');
    writeFileSync(join(dir, 'index.json'), JSON.stringify({ files: {}, chunks: [] }, null, 2), 'utf8');
    writeFileSync(join(dir, 'memory.json'), JSON.stringify([], null, 2), 'utf8');
    return tenant;
  }

  listTenants(): BrainTenant[] {
    const tRoot = join(this.rootDir, 'tenants');
    if (!existsSync(tRoot)) return [];
    const out: BrainTenant[] = [];
    for (const d of readdirSync(tRoot)) {
      const f = join(tRoot, d, 'tenant.json');
      if (existsSync(f)) {
        try {
          out.push(JSON.parse(readFileSync(f, 'utf8')) as BrainTenant);
        } catch {
          // ignore
        }
      }
    }
    return out.sort((a, b) => a.tenantId.localeCompare(b.tenantId));
  }

  getTenant(tenantId: string): BrainTenant | null {
    const f = join(this.tenantDir(tenantId), 'tenant.json');
    return existsSync(f) ? (JSON.parse(readFileSync(f, 'utf8')) as BrainTenant) : null;
  }

  /** Generator: create a fresh isolated session inside a tenant brain. */
  createSession(tenantIdRaw: string, title = 'New Session', customSessionId?: string): BrainSession {
    const tenant = this.getTenant(tenantIdRaw) || this.createTenant(tenantIdRaw);
    const sid = customSessionId ? sanitizeId(customSessionId) : `sess-${Date.now()}-${randomBytes(3).toString('hex')}`;
    const now = new Date().toISOString();
    const session: BrainSession = {
      sessionId: sid,
      tenantId: tenant.tenantId,
      title,
      createdAt: now,
      updatedAt: now,
      messages: [],
    };
    writeFileSync(join(this.tenantDir(tenant.tenantId), 'sessions', `${sid}.json`), JSON.stringify(session, null, 2), 'utf8');
    return session;
  }

  listSessions(tenantId: string): BrainSession[] {
    const sDir = join(this.tenantDir(tenantId), 'sessions');
    if (!existsSync(sDir)) return [];
    const out: BrainSession[] = [];
    for (const f of readdirSync(sDir)) {
      if (!f.endsWith('.json')) continue;
      try {
        out.push(JSON.parse(readFileSync(join(sDir, f), 'utf8')) as BrainSession);
      } catch {
        // ignore
      }
    }
    return out.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  }

  getSession(tenantId: string, sessionId: string): BrainSession | null {
    const f = join(this.tenantDir(tenantId), 'sessions', `${sanitizeId(sessionId)}.json`);
    return existsSync(f) ? (JSON.parse(readFileSync(f, 'utf8')) as BrainSession) : null;
  }

  appendSessionMessage(
    tenantId: string,
    sessionId: string,
    msg: { role: 'user' | 'assistant' | 'system'; content: string; model?: string },
  ): BrainSession {
    const s = this.getSession(tenantId, sessionId) || this.createSession(tenantId, sessionId, sessionId);
    s.messages.push({ ...msg, ts: new Date().toISOString() });
    s.updatedAt = new Date().toISOString();
    writeFileSync(
      join(this.tenantDir(tenantId), 'sessions', `${s.sessionId}.json`),
      JSON.stringify(s, null, 2),
      'utf8',
    );
    return s;
  }

  // ---- Model-Independent Episodic Memory ----

  remember(tenantId: string, text: string, tags: string[] = []): MemoryItem {
    this.createTenant(tenantId);
    const f = join(this.tenantDir(tenantId), 'memory.json');
    const list: MemoryItem[] = existsSync(f) ? JSON.parse(readFileSync(f, 'utf8')) : [];
    const item: MemoryItem = {
      id: `mem-${randomBytes(4).toString('hex')}`,
      text: text.trim(),
      tags: tags.map((t) => t.trim().toLowerCase()).filter(Boolean),
      createdAt: new Date().toISOString(),
      vec: embedTextLocal(text),
    };
    list.unshift(item);
    writeFileSync(f, JSON.stringify(list, null, 2), 'utf8');
    return item;
  }

  listMemory(tenantId: string): MemoryItem[] {
    const f = join(this.tenantDir(tenantId), 'memory.json');
    return existsSync(f) ? (JSON.parse(readFileSync(f, 'utf8')) as MemoryItem[]) : [];
  }

  // ---- Incremental SHA-256 CAS Code Indexer ----

  indexDirectory(tenantId: string, targetDir: string, maxFiles = 400): {
    indexedFiles: number;
    skippedCached: number;
    totalChunks: number;
  } {
    this.createTenant(tenantId, { workspaceDir: targetDir });
    const absRoot = resolve(targetDir);
    const idxFile = join(this.tenantDir(tenantId), 'index.json');
    const prev: { files: Record<string, string>; chunks: CodeChunk[] } = existsSync(idxFile)
      ? JSON.parse(readFileSync(idxFile, 'utf8'))
      : { files: {}, chunks: [] };

    const foundFiles: string[] = [];
    const walk = (d: string) => {
      if (foundFiles.length >= maxFiles) return;
      for (const e of readdirSync(d, { withFileTypes: true })) {
        if (foundFiles.length >= maxFiles) break;
        if (e.isDirectory()) {
          if (!SKIP_DIRS.has(e.name) && !e.name.startsWith('.')) walk(join(d, e.name));
        } else if (INDEXABLE_EXTS.has(extname(e.name).toLowerCase())) {
          foundFiles.push(join(d, e.name));
        }
      }
    };
    walk(absRoot);

    const newFilesMap: Record<string, string> = {};
    const keptChunks = new Map<string, CodeChunk[]>();
    for (const c of prev.chunks) {
      if (!keptChunks.has(c.file)) keptChunks.set(c.file, []);
      keptChunks.get(c.file)!.push(c);
    }

    let indexedFiles = 0;
    let skippedCached = 0;
    const allChunks: CodeChunk[] = [];

    for (const abs of foundFiles) {
      let st;
      try {
        st = statSync(abs);
      } catch {
        continue;
      }
      if (st.size > 256 * 1024) continue; // skip huge generated bundles
      const rel = relative(absRoot, abs).replace(/\\/g, '/');
      const content = readFileSync(abs, 'utf8');
      const hash = createHash('sha256').update(content).digest('hex').slice(0, 16);
      newFilesMap[rel] = hash;

      if (prev.files[rel] === hash && keptChunks.has(rel)) {
        skippedCached++;
        allChunks.push(...keptChunks.get(rel)!);
        continue;
      }

      indexedFiles++;
      const lines = content.split(/\r?\n/);
      const chunkLines = 45;
      for (let start = 0; start < lines.length; start += chunkLines) {
        const slice = lines.slice(start, start + chunkLines).join('\n').trim();
        if (!slice) continue;
        allChunks.push({
          id: `${rel}:${start + 1}`,
          file: rel,
          startLine: start + 1,
          endLine: Math.min(lines.length, start + chunkLines),
          fileHash: hash,
          text: slice.slice(0, 1800),
          vec: embedTextLocal(`${rel}\n${slice}`),
        });
      }
    }

    writeFileSync(idxFile, JSON.stringify({ files: newFilesMap, chunks: allChunks }), 'utf8');
    return { indexedFiles, skippedCached, totalChunks: allChunks.length };
  }

  /** Hybrid BM25/Lexical + Cosine Vector retrieval across code chunks & episodic memory. */
  search(tenantId: string, query: string, topK = 6): SearchHit[] {
    const qVec = embedTextLocal(query);
    const qTerms = query
      .toLowerCase()
      .split(/[^a-z0-9_.-]+/)
      .filter((t) => t.length >= 2);

    const hits: SearchHit[] = [];

    const idxFile = join(this.tenantDir(tenantId), 'index.json');
    if (existsSync(idxFile)) {
      const idx = JSON.parse(readFileSync(idxFile, 'utf8')) as { chunks: CodeChunk[] };
      for (const c of idx.chunks) {
        const score = 0.55 * cosineSim(qVec, c.vec) + 0.45 * lexicalScore(qTerms, `${c.file}\n${c.text}`);
        if (score > 0.08) {
          hits.push({
            source: 'code',
            score: Math.round(score * 1000) / 1000,
            file: c.file,
            startLine: c.startLine,
            endLine: c.endLine,
            text: c.text,
          });
        }
      }
    }

    for (const m of this.listMemory(tenantId)) {
      const score = 0.5 * cosineSim(qVec, m.vec) + 0.5 * lexicalScore(qTerms, `${m.tags.join(' ')} ${m.text}`) + 0.05;
      if (score > 0.1) {
        hits.push({
          source: 'memory',
          score: Math.round(score * 1000) / 1000,
          tags: m.tags,
          text: m.text,
        });
      }
    }

    return hits.sort((a, b) => b.score - a.score).slice(0, topK);
  }
}
