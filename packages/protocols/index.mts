/**
 * @file packages/protocols/index.mts
 * @description Unified Sovereign Agent Protocols:
 *   1. A2A (`createA2aEnvelope`, `verifyA2aHandoff`, `createSwarmChannel` with disjoint `wedge (^)` ownership)
 *   2. A2UI (`renderA2uiTree`, `validateA2uiComponent` for `<b8g-*>` HTML5 Custom Elements)
 *   3. x402 A2A Payments (`X402_BAAS_PROVIDERS`, `createX402PaymentChallenge`, `verifyX402PaymentReceipt`)
 *   4. MCP (`AgentPlatformMcpServer` JSON-RPC 2.0 handler & stdio server with 6 live platform tools)
 *
 * Strictly `erasableSyntaxOnly: true`, Node 22+ built-ins only.
 */

import { randomBytes } from 'node:crypto';
import * as os from 'node:os';
import * as path from 'node:path';
import * as readline from 'node:readline';
import {
  CURATED_MODEL_SPECS,
  getCuratedModelSpec,
  stampModelBlueprint,
} from '../model-ops/index.mts';
import {
  B8gVerifiedMoERouter,
  type CapabilityToken,
  type VerifiedMoeExpert,
} from '../moe-gate/index.mts';
import {
  SubstrateCasStore,
  SubstrateKvStore,
  Zer0qRunner,
  canonicalJson,
  sha512Hex,
  type Zer0qJobManifest,
} from '../zer0q-substrate/index.mts';
import {
  HiKvpArena,
  kg2kvEncodeInto,
  type KgTriple,
} from '../atlas-kv/index.mts';
import {
  detectHardware,
  recommendProfilesAndModels,
} from '../hardware-fit/hardware-fit.mts';

// ============================================================================
// 1. A2A Protocol + Disjoint File-Ownership Wedge (^) Swarm Channel
// ============================================================================

export interface A2aEnvelope<T = unknown> {
  readonly protocol: 'a2a/v1';
  readonly id: string;
  readonly traceId: string;
  readonly fromAgent: string;
  readonly toAgent: string;
  readonly intent: string;
  readonly ownedPaths: readonly string[];
  readonly payload: T;
  readonly createdAt: number;
  readonly expiresAt: number;
  readonly digestSha512: string;
}

export interface CreateA2aEnvelopeOptions<T> {
  readonly fromAgent: string;
  readonly toAgent: string;
  readonly intent: string;
  readonly payload: T;
  readonly ownedPaths?: readonly string[];
  readonly traceId?: string;
  readonly ttlMs?: number;
  readonly nowMs?: number;
}

export interface A2aVerificationResult {
  readonly ok: boolean;
  readonly reason?: string;
}

function computeEnvelopeDigest<T>(env: Omit<A2aEnvelope<T>, 'digestSha512'>): string {
  return sha512Hex(canonicalJson(env));
}

export function createA2aEnvelope<T>(opts: CreateA2aEnvelopeOptions<T>): A2aEnvelope<T> {
  const now = opts.nowMs ?? Date.now();
  const ttlMs = Math.max(1, opts.ttlMs ?? 60_000);
  const id = `a2a-${randomBytes(8).toString('hex')}`;
  const traceId = opts.traceId ?? `trace-${randomBytes(6).toString('hex')}`;
  const base: Omit<A2aEnvelope<T>, 'digestSha512'> = {
    protocol: 'a2a/v1',
    id,
    traceId,
    fromAgent: opts.fromAgent,
    toAgent: opts.toAgent,
    intent: opts.intent,
    ownedPaths: Object.freeze([...(opts.ownedPaths ?? [])]),
    payload: opts.payload,
    createdAt: now,
    expiresAt: now + ttlMs,
  };
  const digestSha512 = computeEnvelopeDigest(base);
  return Object.freeze({ ...base, digestSha512 });
}

export function verifyA2aHandoff<T>(
  envelope: A2aEnvelope<T>,
  opts?: { readonly expectedReceiver?: string; readonly nowMs?: number },
): A2aVerificationResult {
  if (envelope.protocol !== 'a2a/v1') {
    return { ok: false, reason: 'unsupported A2A protocol version' };
  }
  if (!envelope.fromAgent.trim() || !envelope.toAgent.trim() || !envelope.intent.trim()) {
    return { ok: false, reason: 'missing fromAgent, toAgent, or intent' };
  }
  if (opts?.expectedReceiver && envelope.toAgent !== opts.expectedReceiver) {
    return {
      ok: false,
      reason: `receiver mismatch: expected "${opts.expectedReceiver}", got "${envelope.toAgent}"`,
    };
  }
  const now = opts?.nowMs ?? Date.now();
  if (now > envelope.expiresAt) {
    return { ok: false, reason: 'A2A envelope expired' };
  }
  const { digestSha512, ...withoutDigest } = envelope;
  const expectedDigest = computeEnvelopeDigest(withoutDigest);
  if (digestSha512 !== expectedDigest) {
    return { ok: false, reason: 'A2A envelope SHA-512 integrity digest mismatch' };
  }
  return { ok: true };
}

function normalizeGlobOrPath(raw: string): string {
  return raw
    .replace(/\\/g, '/')
    .replace(/\/+$/, '')
    .replace(/\/\*\*$/, '')
    .replace(/\/\*$/, '')
    .toLowerCase();
}

function pathsOverlap(aRaw: string, bRaw: string): boolean {
  const a = normalizeGlobOrPath(aRaw);
  const b = normalizeGlobOrPath(bRaw);
  if (!a || !b) return false;
  if (a === b) return true;
  return a.startsWith(`${b}/`) || b.startsWith(`${a}/`);
}

export class WedgePathCollisionError extends Error {
  public readonly agentA: string;
  public readonly agentB: string;
  public readonly pathA: string;
  public readonly pathB: string;

  constructor(agentA: string, pathA: string, agentB: string, pathB: string) {
    super(`Wedge (^) disjoint path collision between "${agentA}" (${pathA}) and "${agentB}" (${pathB})`);
    this.name = 'WedgePathCollisionError';
    this.agentA = agentA;
    this.agentB = agentB;
    this.pathA = pathA;
    this.pathB = pathB;
  }
}

export function assertDisjointOwnership(
  assignments: ReadonlyArray<{ readonly agentId: string; readonly paths: readonly string[] }>,
): void {
  for (let i = 0; i < assignments.length; i += 1) {
    const a = assignments[i]!;
    for (let j = i + 1; j < assignments.length; j += 1) {
      const b = assignments[j]!;
      for (const pA of a.paths) {
        for (const pB of b.paths) {
          if (pathsOverlap(pA, pB)) {
            throw new WedgePathCollisionError(a.agentId, pA, b.agentId, pB);
          }
        }
      }
    }
  }
}

export interface SwarmChannel {
  readonly channelId: string;
  registerParticipant: (agentId: string, ownedPaths: readonly string[]) => void;
  getParticipants: () => Readonly<Record<string, readonly string[]>>;
  send: <T>(envelope: A2aEnvelope<T>) => A2aVerificationResult;
  subscribe: (agentId: string, listener: (envelope: A2aEnvelope<unknown>) => void) => () => void;
  drain: (agentId: string) => readonly A2aEnvelope<unknown>[];
  createReadableStream: (agentId: string) => ReadableStream<A2aEnvelope<unknown>>;
}

/**
 * Creates an in-memory A2A swarm message channel enforcing `wedge (^)` disjoint file-ownership.
 */
export function createSwarmChannel(opts?: { readonly channelId?: string }): SwarmChannel {
  const channelId = opts?.channelId ?? `swarm-${randomBytes(4).toString('hex')}`;
  const participants = new Map<string, readonly string[]>();
  const mailboxes = new Map<string, A2aEnvelope<unknown>[]>();
  const listeners = new Map<string, Set<(env: A2aEnvelope<unknown>) => void>>();

  const registerParticipant = (agentId: string, ownedPaths: readonly string[]): void => {
    const candidateList = [
      ...Array.from(participants.entries())
        .filter(([id]) => id !== agentId)
        .map(([id, paths]) => ({ agentId: id, paths })),
      { agentId, paths: ownedPaths },
    ];
    assertDisjointOwnership(candidateList);
    participants.set(agentId, Object.freeze([...ownedPaths]));
    if (!mailboxes.has(agentId)) {
      mailboxes.set(agentId, []);
    }
  };

  const getParticipants = (): Readonly<Record<string, readonly string[]>> => {
    const out: Record<string, readonly string[]> = {};
    for (const [id, paths] of participants.entries()) {
      out[id] = paths;
    }
    return Object.freeze(out);
  };

  const send = <T>(envelope: A2aEnvelope<T>): A2aVerificationResult => {
    const check = verifyA2aHandoff(envelope);
    if (!check.ok) {
      return check;
    }
    const senderPaths = participants.get(envelope.fromAgent);
    if (senderPaths && envelope.ownedPaths.length > 0) {
      for (const claimed of envelope.ownedPaths) {
        const covered = senderPaths.some((owned) => pathsOverlap(owned, claimed));
        if (!covered) {
          return {
            ok: false,
            reason: `agent "${envelope.fromAgent}" does not own path "${claimed}"`,
          };
        }
      }
    }
    const queue = mailboxes.get(envelope.toAgent) ?? [];
    queue.push(envelope as A2aEnvelope<unknown>);
    mailboxes.set(envelope.toAgent, queue);

    const subs = listeners.get(envelope.toAgent);
    if (subs) {
      for (const fn of subs) {
        fn(envelope as A2aEnvelope<unknown>);
      }
    }
    return { ok: true };
  };

  const subscribe = (
    agentId: string,
    listener: (envelope: A2aEnvelope<unknown>) => void,
  ): (() => void) => {
    let set = listeners.get(agentId);
    if (!set) {
      set = new Set();
      listeners.set(agentId, set);
    }
    set.add(listener);
    return () => {
      set?.delete(listener);
    };
  };

  const drain = (agentId: string): readonly A2aEnvelope<unknown>[] => {
    const q = mailboxes.get(agentId) ?? [];
    mailboxes.set(agentId, []);
    return Object.freeze(q);
  };

  const createReadableStream = (agentId: string): ReadableStream<A2aEnvelope<unknown>> => {
    let unsub: (() => void) | null = null;
    return new ReadableStream<A2aEnvelope<unknown>>({
      start(controller) {
        const backlog = drain(agentId);
        for (const msg of backlog) {
          controller.enqueue(msg);
        }
        unsub = subscribe(agentId, (msg) => {
          controller.enqueue(msg);
        });
      },
      cancel() {
        if (unsub) {
          unsub();
          unsub = null;
        }
      },
    });
  };

  return Object.freeze({
    channelId,
    registerParticipant,
    getParticipants,
    send,
    subscribe,
    drain,
    createReadableStream,
  });
}

// ============================================================================
// 2. A2UI Protocol (`<b8g-*>` HTML5 Web Components with `data-*` & escaping)
// ============================================================================

export interface A2uiComponentNode {
  readonly tag: string;
  readonly attrs?: Readonly<Record<string, string | number | boolean>>;
  readonly text?: string;
  readonly children?: readonly A2uiComponentNode[];
}

export interface A2uiValidationResult {
  readonly valid: boolean;
  readonly errors: readonly string[];
}

const B8G_CUSTOM_TAG_RE = /^b8g-[a-z][a-z0-9-]*$/;
const SAFE_ATTR_NAME_RE = /^(?:data-[a-z0-9-]+|id|class|slot|role|aria-[a-z0-9-]+|title)$/;

export function escapeHtmlAttr(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

export function escapeHtmlText(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

export function validateA2uiComponent(node: A2uiComponentNode): A2uiValidationResult {
  const errors: string[] = [];
  if (!B8G_CUSTOM_TAG_RE.test(node.tag)) {
    errors.push(`Invalid A2UI tag "${node.tag}": must be an HTML5 custom element starting with "b8g-"`);
  }
  if (node.attrs) {
    for (const [k, v] of Object.entries(node.attrs)) {
      if (!SAFE_ATTR_NAME_RE.test(k)) {
        errors.push(`Disallowed attribute "${k}" on <${node.tag}>: use data-* or safe ARIA/structural attributes`);
      }
      if (typeof v === 'string' && /^\s*javascript:/i.test(v)) {
        errors.push(`Unsafe javascript: URI in attribute "${k}" on <${node.tag}>`);
      }
    }
  }
  if (node.children) {
    for (const child of node.children) {
      const childRes = validateA2uiComponent(child);
      if (!childRes.valid) {
        errors.push(...childRes.errors);
      }
    }
  }
  return Object.freeze({
    valid: errors.length === 0,
    errors: Object.freeze(errors),
  });
}

export function renderA2uiTree(node: A2uiComponentNode): string {
  const validation = validateA2uiComponent(node);
  if (!validation.valid) {
    throw new Error(`Invalid A2UI component tree: ${validation.errors.join('; ')}`);
  }
  const attrParts: string[] = [];
  if (node.attrs) {
    const keys = Object.keys(node.attrs).sort();
    for (const k of keys) {
      const val = node.attrs[k];
      if (val === false || val === undefined) continue;
      if (val === true) {
        attrParts.push(k);
      } else {
        attrParts.push(`${k}="${escapeHtmlAttr(String(val))}"`);
      }
    }
  }
  const openTag = attrParts.length > 0 ? `<${node.tag} ${attrParts.join(' ')}>` : `<${node.tag}>`;
  const innerText = node.text !== undefined ? escapeHtmlText(node.text) : '';
  const innerChildren = node.children ? node.children.map((c) => renderA2uiTree(c)).join('') : '';
  return `${openTag}${innerText}${innerChildren}</${node.tag}>`;
}

// ============================================================================
// 3. x402 BaaS A2A Payments Protocol
// ============================================================================

export interface X402BaasProvider {
  readonly id: string;
  readonly name: string;
  readonly category: 'baas-emi' | 'marketplace-payments' | 'card-issuing' | 'clearing-bank';
  readonly region: string;
  readonly rails: readonly string[];
  readonly currencies: readonly string[];
  readonly specialty: string;
}

export const X402_BAAS_PROVIDERS: readonly X402BaasProvider[] = Object.freeze([
  Object.freeze({
    id: 'solaris',
    name: 'Solaris',
    category: 'baas-emi',
    region: 'Germany / EU',
    rails: Object.freeze(['SEPA', 'SEPA-Instant', 'Cards', 'Digital-Assets']),
    currencies: Object.freeze(['EUR']),
    specialty: 'Full German-licensed BaaS suite including accounts, payments, cards, and custody.',
  }),
  Object.freeze({
    id: 'railsr',
    name: 'Railsr',
    category: 'baas-emi',
    region: 'UK / Lithuania',
    rails: Object.freeze(['FPS', 'SEPA', 'Cards']),
    currencies: Object.freeze(['GBP', 'EUR', 'USD']),
    specialty: 'Embedded finance, virtual accounts, cards, and credit-as-a-service.',
  }),
  Object.freeze({
    id: 'swan',
    name: 'Swan',
    category: 'baas-emi',
    region: 'France / EU',
    rails: Object.freeze(['SEPA', 'SEPA-Instant', 'Cards']),
    currencies: Object.freeze(['EUR']),
    specialty: 'Developer-first European BaaS API for embedded IBAN accounts and payments.',
  }),
  Object.freeze({
    id: 'modulr',
    name: 'Modulr',
    category: 'baas-emi',
    region: 'UK / Ireland / Netherlands',
    rails: Object.freeze(['FPS', 'CHAPS', 'SEPA', 'Bacs']),
    currencies: Object.freeze(['GBP', 'EUR']),
    specialty: 'Automated B2B payment flows, real-time clearing, and high-volume settlement.',
  }),
  Object.freeze({
    id: 'stripe-connect',
    name: 'Stripe Connect',
    category: 'marketplace-payments',
    region: 'Global',
    rails: Object.freeze(['Cards', 'ACH', 'SEPA', 'FPS', 'USDC-x402']),
    currencies: Object.freeze(['USD', 'EUR', 'GBP']),
    specialty: 'Multi-party marketplace payouts, split settlements, and agentic commerce.',
  }),
  Object.freeze({
    id: 'adyen',
    name: 'Adyen',
    category: 'marketplace-payments',
    region: 'Netherlands / Global',
    rails: Object.freeze(['Cards', 'SEPA', 'ACH', 'Local-APM']),
    currencies: Object.freeze(['EUR', 'USD', 'GBP']),
    specialty: 'Enterprise global platform sub-merchant payments and split settlement.',
  }),
  Object.freeze({
    id: 'clearbank',
    name: 'ClearBank',
    category: 'clearing-bank',
    region: 'UK',
    rails: Object.freeze(['FPS', 'CHAPS', 'Bacs', 'SEPA']),
    currencies: Object.freeze(['GBP', 'EUR']),
    specialty: 'Regulated clearing bank providing direct real-time payment rail access.',
  }),
  Object.freeze({
    id: 'marqeta',
    name: 'Marqeta',
    category: 'card-issuing',
    region: 'Global',
    rails: Object.freeze(['JIT-Card-Issuing', 'Virtual-Cards']),
    currencies: Object.freeze(['USD', 'EUR', 'GBP']),
    specialty: 'Just-in-time virtual card issuing and granular per-agent spend controls.',
  }),
]);

export function getX402Provider(idOrName: string): X402BaasProvider | undefined {
  const q = idOrName.trim().toLowerCase();
  return X402_BAAS_PROVIDERS.find(
    (p) => p.id.toLowerCase() === q || p.name.toLowerCase() === q,
  );
}

export interface X402PaymentChallenge {
  readonly status: 402;
  readonly protocol: 'x402-a2a/v1';
  readonly challengeId: string;
  readonly providerId: string;
  readonly recipientAgent: string;
  readonly resource: string;
  readonly amountMinor: number;
  readonly currency: string;
  readonly nonce: string;
  readonly issuedAt: number;
  readonly expiresAt: number;
  readonly challengeDigest: string;
}

export interface X402PaymentReceipt {
  readonly challengeId: string;
  readonly providerId: string;
  readonly payerAgent: string;
  readonly recipientAgent: string;
  readonly amountMinor: number;
  readonly currency: string;
  readonly settlementRef: string;
  readonly settledAt: number;
  readonly receiptProofSha512: string;
}

export interface X402ReceiptVerification {
  readonly valid: boolean;
  readonly reason?: string;
}

export function createX402PaymentChallenge(opts: {
  readonly amountMinor: number;
  readonly currency: string;
  readonly recipientAgent: string;
  readonly providerId: string;
  readonly resource: string;
  readonly ttlMs?: number;
  readonly nowMs?: number;
}): X402PaymentChallenge {
  const provider = getX402Provider(opts.providerId);
  if (!provider) {
    throw new Error(`Unsupported x402 BaaS provider "${opts.providerId}"`);
  }
  if (!Number.isFinite(opts.amountMinor) || opts.amountMinor <= 0) {
    throw new RangeError(`amountMinor must be positive, got ${opts.amountMinor}`);
  }
  const now = opts.nowMs ?? Date.now();
  const base = {
    status: 402 as const,
    protocol: 'x402-a2a/v1' as const,
    challengeId: `x402-${randomBytes(8).toString('hex')}`,
    providerId: provider.id,
    recipientAgent: opts.recipientAgent,
    resource: opts.resource,
    amountMinor: Math.trunc(opts.amountMinor),
    currency: opts.currency.toUpperCase(),
    nonce: randomBytes(12).toString('hex'),
    issuedAt: now,
    expiresAt: now + Math.max(1000, opts.ttlMs ?? 300_000),
  };
  const challengeDigest = sha512Hex(canonicalJson(base));
  return Object.freeze({ ...base, challengeDigest });
}

export function createX402PaymentReceipt(
  challenge: X402PaymentChallenge,
  opts: { readonly payerAgent: string; readonly settlementRef: string; readonly settledAt?: number },
): X402PaymentReceipt {
  const settledAt = opts.settledAt ?? Date.now();
  const proofPayload = {
    challengeId: challenge.challengeId,
    challengeDigest: challenge.challengeDigest,
    providerId: challenge.providerId,
    payerAgent: opts.payerAgent,
    recipientAgent: challenge.recipientAgent,
    amountMinor: challenge.amountMinor,
    currency: challenge.currency,
    settlementRef: opts.settlementRef,
    settledAt,
  };
  return Object.freeze({
    ...proofPayload,
    receiptProofSha512: sha512Hex(canonicalJson(proofPayload)),
  });
}

export function verifyX402PaymentReceipt(
  challenge: X402PaymentChallenge,
  receipt: X402PaymentReceipt,
  nowMs: number = Date.now(),
): X402ReceiptVerification {
  if (receipt.challengeId !== challenge.challengeId) {
    return { valid: false, reason: 'challengeId mismatch' };
  }
  if (receipt.providerId !== challenge.providerId) {
    return { valid: false, reason: 'providerId mismatch' };
  }
  if (receipt.recipientAgent !== challenge.recipientAgent) {
    return { valid: false, reason: 'recipientAgent mismatch' };
  }
  if (receipt.currency.toUpperCase() !== challenge.currency.toUpperCase()) {
    return { valid: false, reason: 'currency mismatch' };
  }
  if (receipt.amountMinor < challenge.amountMinor) {
    return { valid: false, reason: 'insufficient payment amount' };
  }
  if (nowMs > challenge.expiresAt || receipt.settledAt > challenge.expiresAt) {
    return { valid: false, reason: 'x402 challenge expired' };
  }
  const expectedProof = sha512Hex(
    canonicalJson({
      challengeId: challenge.challengeId,
      challengeDigest: challenge.challengeDigest,
      providerId: challenge.providerId,
      payerAgent: receipt.payerAgent,
      recipientAgent: challenge.recipientAgent,
      amountMinor: receipt.amountMinor,
      currency: receipt.currency,
      settlementRef: receipt.settlementRef,
      settledAt: receipt.settledAt,
    }),
  );
  if (receipt.receiptProofSha512 !== expectedProof) {
    return { valid: false, reason: 'x402 receipt SHA-512 proof mismatch' };
  }
  return { valid: true };
}

// ============================================================================
// 4. MCP Server (JSON-RPC 2.0 + stdio exposing 6 live Agent Platform tools)
// ============================================================================

export interface McpToolDefinition {
  readonly name: string;
  readonly description: string;
  readonly inputSchema: Readonly<Record<string, unknown>>;
}

export interface JsonRpcRequest {
  readonly jsonrpc: '2.0';
  readonly id?: string | number | null;
  readonly method: string;
  readonly params?: Readonly<Record<string, unknown>>;
}

export interface JsonRpcResponse {
  readonly jsonrpc: '2.0';
  readonly id: string | number | null;
  readonly result?: unknown;
  readonly error?: { readonly code: number; readonly message: string; readonly data?: unknown };
}

export const AGENT_PLATFORM_MCP_TOOLS: readonly McpToolDefinition[] = Object.freeze([
  {
    name: 'agent.model.catalog',
    description: 'Lists curated sovereign model architectures or fetches a specific model spec by id.',
    inputSchema: {
      type: 'object',
      properties: { id: { type: 'string' } },
    },
  },
  {
    name: 'agent.model.stamp',
    description: 'Stamps a ModelOpSpec into a 64-byte-aligned B8G_ARENA_V1 blueprint.',
    inputSchema: {
      type: 'object',
      required: ['id'],
      properties: { id: { type: 'string' } },
    },
  },
  {
    name: 'agent.moe.verify',
    description: 'Runs a boundary-verified MoE routing step with ingress/egress L2 bounds and CapabilityToken.',
    inputSchema: {
      type: 'object',
      required: ['vector'],
      properties: {
        vector: { type: 'array', items: { type: 'number' } },
        tenantId: { type: 'string' },
        allowedScopes: { type: 'array', items: { type: 'string' } },
      },
    },
  },
  {
    name: 'agent.zer0q.lookup',
    description: 'Runs SKIMMER < 2 ms CAS short-circuit lookup for a zer0q.job.v1 manifest.',
    inputSchema: {
      type: 'object',
      required: ['manifest'],
      properties: { manifest: { type: 'object' } },
    },
  },
  {
    name: 'agent.atlaskv.query',
    description: 'Queries an AtlasKV HiKVP arena over knowledge-graph triples with tenant scope isolation.',
    inputSchema: {
      type: 'object',
      required: ['query', 'triples'],
      properties: {
        query: { type: 'string' },
        triples: { type: 'array' },
        allowedScopes: { type: 'array', items: { type: 'string' } },
        topK: { type: 'number' },
      },
    },
  },
  {
    name: 'agent.hardware.fit',
    description: 'Detects host hardware metrics (RAM, CPU cores, free disk) and recommends fitting profiles/models.',
    inputSchema: {
      type: 'object',
      properties: { targetDir: { type: 'string' } },
    },
  },
]);

export class AgentPlatformMcpServer {
  public readonly serverInfo = Object.freeze({
    name: '@awesome-os/agent-platform-mcp',
    version: '1.0.0',
  });
  public readonly zer0qRunner: Zer0qRunner;

  constructor(opts?: { readonly zer0qRunner?: Zer0qRunner; readonly casRoot?: string }) {
    if (opts?.zer0qRunner) {
      this.zer0qRunner = opts.zer0qRunner;
    } else {
      const casDir = opts?.casRoot ?? path.join(os.tmpdir(), 'b8g-mcp-zer0q-cas');
      this.zer0qRunner = new Zer0qRunner(new SubstrateCasStore(casDir), new SubstrateKvStore());
    }
  }

  public listTools(): readonly McpToolDefinition[] {
    return AGENT_PLATFORM_MCP_TOOLS;
  }

  public async callTool(
    name: string,
    args: Readonly<Record<string, unknown>> = {},
  ): Promise<unknown> {
    switch (name) {
      case 'agent.model.catalog': {
        const id = typeof args.id === 'string' ? args.id : undefined;
        if (id) {
          const spec = getCuratedModelSpec(id);
          if (!spec) throw new Error(`Unknown model spec id "${id}"`);
          return { spec };
        }
        return { models: CURATED_MODEL_SPECS };
      }

      case 'agent.model.stamp': {
        const id = String(args.id ?? '');
        const spec = getCuratedModelSpec(id);
        if (!spec) throw new Error(`Unknown model spec id "${id}"`);
        return stampModelBlueprint(spec);
      }

      case 'agent.moe.verify': {
        const rawVec = Array.isArray(args.vector) ? args.vector.map(Number) : [];
        const hiddenSize = Math.max(1, rawVec.length);
        const input = new Float32Array(rawVec);
        const router = new B8gVerifiedMoERouter({
          hiddenSize,
          topK: 2,
          loopPasses: 1,
          maxIngressL2Norm: 1e4,
          maxEgressL2Norm: 1e4,
          loopBlendAlpha: 0.5,
        });
        const expertA: VerifiedMoeExpert = {
          id: 1,
          name: 'public-solver',
          requiredScope: 'moe.public',
          routerKey: new Float32Array(hiddenSize).fill(1),
          forward: (inp, out) => {
            for (let i = 0; i < hiddenSize; i += 1) out[i] = inp[i]! * 0.5;
          },
        };
        const expertB: VerifiedMoeExpert = {
          id: 2,
          name: 'private-auditor',
          requiredScope: 'moe.private',
          routerKey: new Float32Array(hiddenSize).fill(0.5),
          forward: (inp, out) => {
            for (let i = 0; i < hiddenSize; i += 1) out[i] = inp[i]! * 0.8;
          },
        };
        router.registerExpert(expertA);
        router.registerExpert(expertB);
        const token: CapabilityToken = {
          tenantId: typeof args.tenantId === 'string' ? args.tenantId : 'local-user',
          allowedScopes: Array.isArray(args.allowedScopes)
            ? args.allowedScopes.map(String)
            : ['moe.public'],
        };
        const result = router.dispatch(input, token);
        return {
          ok: result.ok,
          passesExecuted: result.passesExecuted,
          activeExpertsExecuted: result.activeExpertsExecuted,
          quarantinedCount: result.quarantinedCount,
          capabilityDenials: result.capabilityDenials,
          output: Array.from(result.output),
          audits: result.audits,
        };
      }

      case 'agent.zer0q.lookup': {
        const manifest = args.manifest as Zer0qJobManifest | undefined;
        if (!manifest || manifest.apiVersion !== 'zer0q.awesome-os/v1') {
          throw new Error('Invalid zer0q.awesome-os/v1 manifest');
        }
        return this.zer0qRunner.lookup(manifest);
      }

      case 'agent.atlaskv.query': {
        const query = String(args.query ?? '');
        const triples = Array.isArray(args.triples) ? (args.triples as KgTriple[]) : [];
        const dim = 32;
        const arena = HiKvpArena.fromTriples(triples, dim, Math.min(8, Math.max(1, triples.length)));
        const qVec = new Float32Array(dim);
        kg2kvEncodeInto(query, dim, qVec, 0);
        const allowed = Array.isArray(args.allowedScopes)
          ? new Set(args.allowedScopes.map(String))
          : undefined;
        const topK = typeof args.topK === 'number' ? Math.max(1, args.topK) : 3;
        return {
          hits: arena.queryHiKvp(qVec, 2, topK, allowed),
        };
      }

      case 'agent.hardware.fit': {
        const targetDir = typeof args.targetDir === 'string' ? args.targetDir : undefined;
        const hardware = detectHardware(targetDir);
        const recommendations = recommendProfilesAndModels(hardware);
        return { hardware, recommendations };
      }

      default:
        throw new Error(`Unknown MCP tool "${name}"`);
    }
  }

  public async handleRequest(req: JsonRpcRequest): Promise<JsonRpcResponse> {
    const id = req.id ?? null;
    try {
      switch (req.method) {
        case 'initialize':
          return {
            jsonrpc: '2.0',
            id,
            result: {
              protocolVersion: '2024-11-05',
              capabilities: { tools: {} },
              serverInfo: this.serverInfo,
            },
          };

        case 'ping':
          return { jsonrpc: '2.0', id, result: {} };

        case 'tools/list':
          return {
            jsonrpc: '2.0',
            id,
            result: { tools: this.listTools() },
          };

        case 'tools/call': {
          const toolName = String(req.params?.name ?? '');
          const toolArgs = (req.params?.arguments ?? {}) as Readonly<Record<string, unknown>>;
          const output = await this.callTool(toolName, toolArgs);
          return {
            jsonrpc: '2.0',
            id,
            result: {
              content: [{ type: 'text', text: JSON.stringify(output) }],
              structuredContent: output,
            },
          };
        }

        default:
          return {
            jsonrpc: '2.0',
            id,
            error: { code: -32601, message: `Method not found: ${req.method}` },
          };
      }
    } catch (err) {
      return {
        jsonrpc: '2.0',
        id,
        error: {
          code: -32603,
          message: err instanceof Error ? err.message : String(err),
        },
      };
    }
  }

  public async handleLine(line: string): Promise<string | null> {
    const trimmed = line.trim();
    if (!trimmed) return null;
    try {
      const req = JSON.parse(trimmed) as JsonRpcRequest;
      const res = await this.handleRequest(req);
      return JSON.stringify(res);
    } catch (err) {
      const res: JsonRpcResponse = {
        jsonrpc: '2.0',
        id: null,
        error: {
          code: -32700,
          message: err instanceof Error ? err.message : 'Parse error',
        },
      };
      return JSON.stringify(res);
    }
  }

  public serveStdio(
    input: NodeJS.ReadableStream = process.stdin,
    output: NodeJS.WritableStream = process.stdout,
  ): readline.Interface {
    const rl = readline.createInterface({ input, crlfDelay: Infinity });
    rl.on('line', (line) => {
      void this.handleLine(line).then((resp) => {
        if (resp !== null) {
          output.write(`${resp}\n`);
        }
      });
    });
    return rl;
  }
}
