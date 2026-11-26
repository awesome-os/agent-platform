/**
 * B8G Universal Brain — Model-Agnostic OpenRouter / Ollama / OpenAI-Compatible Router.
 *
 * Retrieves top-K code chunks + episodic memory from `B8gBrainStore`, builds a tight
 * context window, and routes to OpenRouter, local Ollama, or any OpenAI-compatible API.
 */
import type { B8gBrainStore, SearchHit } from './brain-store.mts';

export interface BrainChatRequest {
  tenantId: string;
  sessionId: string;
  prompt: string;
  model?: string;
  providerUrl?: string;
  apiKey?: string;
  topK?: number;
  /** When true (or when no API key is set for remote OpenRouter), returns assembled context without external call. */
  dryRun?: boolean;
}

export interface BrainChatResponse {
  tenantId: string;
  sessionId: string;
  model: string;
  retrieved: SearchHit[];
  reply: string;
  dryRun: boolean;
}

export async function runBrainChat(
  store: B8gBrainStore,
  req: BrainChatRequest,
): Promise<BrainChatResponse> {
  const tenant = store.getTenant(req.tenantId) || store.createTenant(req.tenantId);
  const session = store.getSession(tenant.tenantId, req.sessionId) || store.createSession(tenant.tenantId, req.sessionId, req.sessionId);

  const retrieved = store.search(tenant.tenantId, req.prompt, req.topK ?? 5);
  const contextBlocks = retrieved.map((h, i) =>
    h.source === 'memory'
      ? `[Memory #${i + 1} score=${h.score} tags=${(h.tags || []).join(',')}]\n${h.text}`
      : `[Code #${i + 1} ${h.file}:${h.startLine}-${h.endLine} score=${h.score}]\n${h.text}`,
  );

  const systemPrompt = [
    `You are B8G-Brain serving tenant "${tenant.tenantId}" (session "${session.sessionId}").`,
    `Answer concisely and grounded strictly in the retrieved workspace code and episodic memory below.`,
    contextBlocks.length ? `\n--- RETRIEVED CONTEXT ---\n${contextBlocks.join('\n\n')}` : '',
  ].join('\n');

  const model = req.model || tenant.defaultModel || 'openrouter/auto';
  const providerUrl = req.providerUrl || process.env.B8G_BRAIN_PROVIDER_URL || tenant.providerUrl;
  const apiKey = req.apiKey || process.env.OPENROUTER_API_KEY || process.env.OPENAI_API_KEY || '';
  const isLocal = /^https?:\/\/(127\.0\.0\.1|localhost)/i.test(providerUrl);

  store.appendSessionMessage(tenant.tenantId, session.sessionId, { role: 'user', content: req.prompt, model });

  if (req.dryRun || (!apiKey && !isLocal)) {
    const reply =
      `[B8G-Brain context-ready | model=${model} | retrieved=${retrieved.length} items]\n` +
      `Set OPENROUTER_API_KEY (or point providerUrl to local Ollama http://127.0.0.1:11434/v1/chat/completions) for live generation.\n` +
      `Top matches: ${retrieved.map((r) => (r.file ? `${r.file}:${r.startLine}` : `memory(${r.tags?.join(',')})`)).join(', ') || 'none'}`;
    store.appendSessionMessage(tenant.tenantId, session.sessionId, { role: 'assistant', content: reply, model });
    return { tenantId: tenant.tenantId, sessionId: session.sessionId, model, retrieved, reply, dryRun: true };
  }

  const messages = [
    { role: 'system', content: systemPrompt },
    ...session.messages.slice(-10).map((m) => ({ role: m.role, content: m.content })),
  ];

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'HTTP-Referer': 'https://awesome-os.local',
    'X-Title': 'B8G-Brain Universal IDE Brain',
  };
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

  const res = await fetch(providerUrl, {
    method: 'POST',
    headers,
    body: JSON.stringify({ model, messages }),
  });

  if (!res.ok) {
    throw new Error(`Provider ${providerUrl} returned HTTP ${res.status}: ${await res.text()}`);
  }

  const data = (await res.json()) as { choices?: Array<{ message?: { content?: string } }> };
  const reply = data?.choices?.[0]?.message?.content || '(empty response)';
  store.appendSessionMessage(tenant.tenantId, session.sessionId, { role: 'assistant', content: reply, model });
  return { tenantId: tenant.tenantId, sessionId: session.sessionId, model, retrieved, reply, dryRun: false };
}
