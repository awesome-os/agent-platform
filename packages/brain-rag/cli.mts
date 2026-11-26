#!/usr/bin/env node
/**
 * B8G Universal Sovereign Brain CLI & HTTP Server (`.\bin\b8g-brain`).
 */
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { B8gBrainStore } from './brain-store.mts';
import { runBrainChat } from './brain-router.mts';

async function readJson(req: IncomingMessage): Promise<any> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(Buffer.from(c));
  return chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : {};
}

export class B8gBrainServer {
  private server: Server;
  readonly store: B8gBrainStore;
  private host: string;
  private port: number;

  constructor(opts: { rootDir?: string; host?: string; port?: number } = {}) {
    this.store = new B8gBrainStore(opts.rootDir);
    this.host = opts.host ?? '127.0.0.1';
    this.port = opts.port ?? 7440;
    this.server = createServer((req, res) => void this.handle(req, res));
  }

  get url(): string {
    const addr = this.server.address();
    const port = typeof addr === 'object' && addr ? addr.port : this.port;
    return `http://${this.host}:${port}`;
  }

  async start(): Promise<string> {
    await new Promise<void>((res, rej) => {
      this.server.once('error', rej);
      this.server.listen(this.port, this.host, () => res());
    });
    return this.url;
  }

  async stop(): Promise<void> {
    await new Promise<void>((res) => this.server.close(() => res()));
  }

  private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }

    const u = new URL(req.url || '/', this.url);
    const p = decodeURIComponent(u.pathname);

    try {
      if (req.method === 'GET' && p === '/healthz') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true, tenants: this.store.listTenants().length }));
        return;
      }

      if (p === '/api/tenants') {
        if (req.method === 'GET') {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ tenants: this.store.listTenants() }, null, 2));
          return;
        }
        if (req.method === 'POST') {
          const b = await readJson(req);
          const t = this.store.createTenant(String(b.tenantId || b.name || ''), b);
          res.writeHead(201, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(t, null, 2));
          return;
        }
      }

      const m = p.match(/^\/api\/tenants\/([^/]+)\/(sessions|index|search|memory|chat)$/);
      if (m) {
        const [, tenantId, action] = m;
        if (action === 'sessions') {
          if (req.method === 'GET') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ sessions: this.store.listSessions(tenantId) }, null, 2));
            return;
          }
          if (req.method === 'POST') {
            const b = await readJson(req);
            const s = this.store.createSession(tenantId, b.title || 'New Session', b.sessionId);
            res.writeHead(201, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(s, null, 2));
            return;
          }
        }
        if (action === 'index' && req.method === 'POST') {
          const b = await readJson(req);
          const stat = this.store.indexDirectory(tenantId, String(b.dir || process.cwd()));
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(stat, null, 2));
          return;
        }
        if (action === 'search' && req.method === 'GET') {
          const q = u.searchParams.get('q') || '';
          const k = Number(u.searchParams.get('k') || 6);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ query: q, hits: this.store.search(tenantId, q, k) }, null, 2));
          return;
        }
        if (action === 'memory') {
          if (req.method === 'GET') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ memory: this.store.listMemory(tenantId) }, null, 2));
            return;
          }
          if (req.method === 'POST') {
            const b = await readJson(req);
            const item = this.store.remember(tenantId, String(b.text || ''), b.tags || []);
            res.writeHead(201, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(item, null, 2));
            return;
          }
        }
        if (action === 'chat' && req.method === 'POST') {
          const b = await readJson(req);
          const out = await runBrainChat(this.store, {
            tenantId,
            sessionId: String(b.sessionId || 'default'),
            prompt: String(b.prompt || ''),
            model: b.model,
            providerUrl: b.providerUrl,
            apiKey: b.apiKey,
            dryRun: b.dryRun,
          });
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(out, null, 2));
          return;
        }
      }

      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Not Found' }));
    } catch (err: any) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err?.message || String(err) }));
    }
  }
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  const args = process.argv.slice(2);
  const cmd = args[0] || 'help';
  const rootDir = process.env.B8G_BRAIN_ROOT || 'E:\\b8g-brains';
  const store = new B8gBrainStore(rootDir);

  if (cmd === 'init-tenant' && args[1]) {
    const wsIdx = args.indexOf('--workspace');
    const t = store.createTenant(args[1], { workspaceDir: wsIdx >= 0 ? args[wsIdx + 1] : undefined });
    console.log(JSON.stringify(t, null, 2));
  } else if (cmd === 'new-session' && args[1]) {
    const titleIdx = args.indexOf('--title');
    const s = store.createSession(args[1], titleIdx >= 0 ? args[titleIdx + 1] : 'Fresh Session');
    console.log(JSON.stringify(s, null, 2));
  } else if (cmd === 'index' && args[1]) {
    const stat = store.indexDirectory(args[1], args[2] || process.cwd());
    console.log(JSON.stringify(stat, null, 2));
  } else if (cmd === 'search' && args[1] && args[2]) {
    console.log(JSON.stringify(store.search(args[1], args[2]), null, 2));
  } else if (cmd === 'remember' && args[1] && args[2]) {
    console.log(JSON.stringify(store.remember(args[1], args[2], (args[3] || '').split(',')), null, 2));
  } else if (cmd === 'serve') {
    const pIdx = args.indexOf('--port');
    const port = pIdx >= 0 && args[pIdx + 1] ? Number(args[pIdx + 1]) : 7440;
    const srv = new B8gBrainServer({ rootDir, port });
    const url = await srv.start();
    console.log(`[b8g-brain] Universal Sovereign Brain API listening on ${url}`);
  } else {
    console.log(
      [
        'B8G Universal Sovereign Brain (Multi-Tenant + Multi-Session + OpenRouter/Ollama)',
        '  b8g-brain init-tenant <tenant> [--workspace <dir>]',
        '  b8g-brain new-session <tenant> [--title "..."]',
        '  b8g-brain index       <tenant> <dir>',
        '  b8g-brain search      <tenant> "<query>"',
        '  b8g-brain remember    <tenant> "<fact>" [tag1,tag2]',
        '  b8g-brain serve       [--port 7440]',
      ].join('\n'),
    );
  }
}
