/**
 * @file intersections/b8g-devops/index.mts
 * @description Centralized single-source-of-truth DevOps & Emergency-Worker Routing unit
 * for B8G / StealifyLang.
 *
 * - Strictly `erasableSyntaxOnly: true` and zero external / `node:*` dependencies.
 * - Declarative `ExecutionContext` (`persona: 'devops'`) and SAB-ring Sections.
 * - Persona-gated emission: normal `consumer` callers incur ZERO ring writes,
 *   ZERO `console.warn` spam, and ZERO emergency-worker dispatch overhead.
 * - Single-claim dispatch per `EmergencyWorkerKind` so multi-agent swarms never
 *   duplicate firefighting on the same incident plane.
 */

import {
  defineSection,
  f,
  intersect,
  isDevOpsAttached,
  s,
  shouldEmitForPersona,
  type ExecutionContext,
  type PersonaPlane,
} from '../schema/index.mts';

// ============================================================================
// 1. Declarative StealifyLang DevOps ExecutionContext & Section Contracts
// ============================================================================

export const B8G_DEVOPS_CONTEXT: ExecutionContext = Object.freeze({
  boundary: 'in-process',
  display: 'headless',
  lifetime: 'singleton-service',
  transport: 'os-pipe',
  capability: Object.freeze({
    tenantId: 'local-user',
    scopes: Object.freeze(['devops.observe']),
    epoch: 1,
    privateData: true,
  }),
  locality: 'local-arena',
  persona: 'devops',
});

export const B8gDevOpsTelemetrySection = defineSection({
  id: 'b8g/devops-telemetry',
  version: '1.0.0',
  substrate: s.sabRing({ ringBytes: 65536 }),
  context: B8G_DEVOPS_CONTEXT,
  fields: {
    totalEmitted: f.int32().default(0),
    totalDroppedForConsumer: f.int32().default(0),
    ringSize: f.int32().default(0),
    activeWorkers: f.int32().default(0),
    lastEventTs: f.timestamp().default(0),
  },
});

export const B8gDevOpsAdvisorySection = defineSection({
  id: 'b8g/devops-advisories',
  version: '1.0.0',
  substrate: s.sabRing({ ringBytes: 65536 }),
  context: B8G_DEVOPS_CONTEXT,
  fields: {
    lastEventId: f.string({ maxBytes: 64 }).default(''),
    lastSeverity: f.string({ maxBytes: 8 }).default(''),
    lastDomain: f.string({ maxBytes: 32 }).default(''),
    lastWorkerTarget: f.string({ maxBytes: 32 }).default(''),
    lastSource: f.string({ maxBytes: 128 }).default(''),
    lastCode: f.string({ maxBytes: 64 }).default(''),
    lastMessage: f.string({ maxBytes: 512 }).default(''),
  },
});

export const B8gDevOpsIntersection = intersect('b8g/devops', [
  B8gDevOpsTelemetrySection,
  B8gDevOpsAdvisorySection,
] as const);

// ============================================================================
// 2. Structured Emergency Routing Types & Deterministic Resolver
// ============================================================================

export type DevOpsSeverity = 'P0' | 'P1' | 'P2' | 'P3';

export type DevOpsDomain =
  | 'security'
  | 'sre'
  | 'dev'
  | 'telemetry'
  | 'context-algebra';

export type EmergencyWorkerKind =
  | 'security-containment'
  | 'sre-self-heal'
  | 'dev-error-loop'
  | 'telemetry-capacity';

export interface DevOpsEvent {
  readonly id: string;
  readonly ts: number;
  readonly severity: DevOpsSeverity;
  readonly domain: DevOpsDomain;
  readonly source: string;
  readonly code: string;
  readonly message: string;
  readonly minPersona: PersonaPlane;
  readonly workerTarget: EmergencyWorkerKind;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

export type EmergencyWorkerHandler = (event: DevOpsEvent) => void;

export interface EmitDevOpsEventInput {
  readonly severity: DevOpsSeverity;
  readonly domain: DevOpsDomain;
  readonly source: string;
  readonly code: string;
  readonly message: string;
  readonly minPersona: PersonaPlane;
  readonly workerTarget?: EmergencyWorkerKind;
  readonly metadata?: Readonly<Record<string, unknown>>;
  readonly emitConsole?: boolean;
}

export interface DevOpsSnapshot {
  readonly totalEmitted: number;
  readonly totalDroppedForConsumer: number;
  readonly ringSize: number;
  readonly events: readonly DevOpsEvent[];
  readonly registeredWorkers: readonly EmergencyWorkerKind[];
}

export interface DebugSymbolEntry {
  readonly file: string;
  readonly line: number;
  readonly fn: string;
}

export interface DebugSidecarManifest {
  readonly buildId: string;
  readonly componentId: string;
  readonly abiVersion: string;
  readonly symbols: Readonly<Record<string, DebugSymbolEntry>>;
}

/**
 * Deterministically maps `(severity, domain)` to the single responsible
 * emergency worker kind.
 */
export function resolveEmergencyWorker(
  severity: DevOpsSeverity,
  domain: DevOpsDomain,
): EmergencyWorkerKind {
  switch (domain) {
    case 'security':
      return 'security-containment';
    case 'sre':
      return 'sre-self-heal';
    case 'dev':
    case 'context-algebra':
      return 'dev-error-loop';
    case 'telemetry':
      return 'telemetry-capacity';
  }
  switch (severity) {
    case 'P0':
      return 'security-containment';
    case 'P1':
      return 'sre-self-heal';
    case 'P2':
      return 'dev-error-loop';
    case 'P3':
      return 'telemetry-capacity';
  }
}

// ============================================================================
// 3. Centralized DevOps Hub Singleton (Bounded Ring + Single-Claim Dispatch)
// ============================================================================

const MAX_RING_EVENTS = 256;

const ringBuffer: DevOpsEvent[] = [];
const workerRegistry = new Map<EmergencyWorkerKind, EmergencyWorkerHandler>();
const debugSidecarRegistry = new Map<string, DebugSidecarManifest>();
let totalEmittedCount = 0;
let totalDroppedForConsumerCount = 0;
let eventSequence = 0;

/**
 * Registers a paired `.debug.b8p` sidecar manifest keyed by `manifest.buildId`
 * for direct on-demand DevOps symbol resolution.
 */
export function registerDebugSidecar(manifest: DebugSidecarManifest): void {
  debugSidecarRegistry.set(manifest.buildId, manifest);
}

/**
 * Persona-gated direct on-demand debug symbol lookup for DevOps / kernel-audit.
 *
 * - `consumer` callers (and omitted `callerCtx`) ALWAYS receive `null` immediately
 *   with zero sidecar map lookup overhead.
 * - `devops` and `kernel-audit` callers receive bit-exact symbol resolution.
 */
export function lookupDebugSymbolForDevOps(
  buildId: string,
  addrHex: string,
  callerCtx?: ExecutionContext,
): (DebugSymbolEntry & { readonly buildId: string; readonly componentId: string }) | null {
  const callerPersona: PersonaPlane = callerCtx?.persona ?? 'consumer';
  if (!shouldEmitForPersona('devops', callerPersona)) {
    return null;
  }

  const manifest = debugSidecarRegistry.get(buildId);
  if (!manifest) {
    return null;
  }

  const entry = Object.prototype.hasOwnProperty.call(manifest.symbols, addrHex)
    ? manifest.symbols[addrHex]
    : undefined;
  if (!entry) {
    return null;
  }

  return Object.freeze({
    file: entry.file,
    line: entry.line,
    fn: entry.fn,
    buildId: manifest.buildId,
    componentId: manifest.componentId,
  });
}

/**
 * Registers the single active emergency worker handler for `kind`
 * (single-claim dispatch: replacing any prior claimant prevents duplicate
 * multi-agent firefighting). Returns an idempotent disposer.
 */
export function registerEmergencyWorker(
  kind: EmergencyWorkerKind,
  handler: EmergencyWorkerHandler,
): () => void {
  workerRegistry.set(kind, handler);
  return () => {
    if (workerRegistry.get(kind) === handler) {
      workerRegistry.delete(kind);
    }
  };
}

/**
 * Persona-gated DevOps event emitter.
 *
 * - If `shouldEmitForPersona(input.minPersona, callerCtx?.persona ?? 'consumer')`
 *   is `false`, returns `false` immediately with ZERO ring writes, ZERO
 *   `console.warn`, and ZERO emergency-worker dispatch.
 * - Otherwise appends to the bounded 256-event ring buffer, dispatches to the
 *   single claimed `EmergencyWorkerHandler` for `workerTarget`, and optionally
 *   emits to `console.warn` only when `emitConsole !== false` and
 *   `isDevOpsAttached(callerCtx)`.
 */
export function emitDevOpsEvent(
  input: EmitDevOpsEventInput,
  callerCtx?: ExecutionContext,
): boolean {
  const callerPersona: PersonaPlane = callerCtx?.persona ?? 'consumer';
  if (!shouldEmitForPersona(input.minPersona, callerPersona)) {
    totalDroppedForConsumerCount += 1;
    return false;
  }

  eventSequence += 1;
  const workerTarget =
    input.workerTarget ?? resolveEmergencyWorker(input.severity, input.domain);

  const event: DevOpsEvent = Object.freeze({
    id: `devops-${eventSequence}`,
    ts: Date.now(),
    severity: input.severity,
    domain: input.domain,
    source: input.source,
    code: input.code,
    message: input.message,
    minPersona: input.minPersona,
    workerTarget,
    ...(input.metadata !== undefined ? { metadata: input.metadata } : {}),
  });

  if (ringBuffer.length >= MAX_RING_EVENTS) {
    ringBuffer.shift();
  }
  ringBuffer.push(event);
  totalEmittedCount += 1;

  const handler = workerRegistry.get(workerTarget);
  if (handler) {
    handler(event);
  }

  if (
    input.emitConsole !== false &&
    callerCtx !== undefined &&
    isDevOpsAttached(callerCtx)
  ) {
    console.warn(
      `[B8G DevOps][${event.severity}][${event.domain}->${event.workerTarget}] ${event.source} (${event.code}): ${event.message}`,
    );
  }

  return true;
}

/**
 * Returns an immutable point-in-time snapshot of the DevOps hub state.
 */
export function getDevOpsSnapshot(): DevOpsSnapshot {
  return Object.freeze({
    totalEmitted: totalEmittedCount,
    totalDroppedForConsumer: totalDroppedForConsumerCount,
    ringSize: ringBuffer.length,
    events: Object.freeze([...ringBuffer]),
    registeredWorkers: Object.freeze(Array.from(workerRegistry.keys())),
  });
}

/**
 * Clears all ring events, counters, registered emergency workers, and debug sidecars.
 */
export function clearDevOpsHub(): void {
  ringBuffer.length = 0;
  workerRegistry.clear();
  debugSidecarRegistry.clear();
  totalEmittedCount = 0;
  totalDroppedForConsumerCount = 0;
  eventSequence = 0;
}
