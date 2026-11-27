/**
 * @file packages/zer0q-substrate/deadhand.mts
 * @description DeadhandController: autonomous `cron`/`webhook`/`cas-diff` triggers
 * with KV idempotency keys (`idem:<trigger>:<digest>`), atomic `alias` set/get/rollback,
 * TTL lease heartbeat + orphan reaper, and deterministic `replay(runId, overrideModel?)`.
 *
 * Strictly `erasableSyntaxOnly: true`, Node 22+ built-ins only.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import {
  type DeadhandFireResult,
  type DeadhandTriggerDef,
  DeadhandTriggerRegistry,
  type LeaseRecord,
  SubstrateCasStore,
  SubstrateKvStore,
  type Zer0qExecutionContext,
  type Zer0qExecutorOutput,
  type Zer0qJobManifest,
  type Zer0qReplayResult,
  type Zer0qReplaySnapshot,
  Zer0qRunner,
  computeInputDigest,
} from './index.mts';

export interface AliasRevision {
  readonly name: string;
  readonly casHash: string;
  readonly updatedAt: number;
}

export interface AliasMutationResult {
  readonly ok: boolean;
  readonly name: string;
  readonly previousHash?: string;
  readonly currentHash?: string;
  readonly rolledBack?: boolean;
}

export interface CasDiffPayload {
  readonly prevTreeDigest: string;
  readonly nextTreeDigest: string;
  readonly changedFiles?: readonly string[];
}

export interface DeadhandCasDiffResult<T = unknown> {
  readonly skippedUnchanged: boolean;
  readonly fireResult?: DeadhandFireResult<T>;
}

export interface PersistedDeadhandState {
  readonly aliases: Readonly<Record<string, string>>;
  readonly aliasHistory: Readonly<Record<string, readonly string[]>>;
  readonly leases: readonly LeaseRecord[];
  readonly triggers: readonly DeadhandTriggerDef[];
}

/**
 * High-level DEADHAND orchestration controller combining:
 *  - `cron` / `webhook` / `cas-diff` / `git` triggers with `idem:<trigger>:<digest>` dedup
 *  - Atomic `alias` set / get / rollback with `compareAndSwap` & revision history
 *  - TTL lease heartbeat & automatic orphan reaper
 *  - Deterministic `replay(runId, overrideModel?)`
 */
export interface DeadhandControllerOptions {
  readonly cas: SubstrateCasStore;
  readonly kv?: SubstrateKvStore;
  readonly runner?: Zer0qRunner;
}

export class DeadhandController {
  public readonly cas: SubstrateCasStore;
  public readonly kv: SubstrateKvStore;
  public readonly runner: Zer0qRunner;
  public readonly registry: DeadhandTriggerRegistry;
  private readonly aliasHistory = new Map<string, string[]>();
  private reaperTimer: ReturnType<typeof setInterval> | null = null;

  constructor(opts: DeadhandControllerOptions) {
    this.cas = opts.cas;
    this.kv = opts.kv ?? new SubstrateKvStore();
    this.runner = opts.runner ?? new Zer0qRunner(this.cas, this.kv);
    this.registry = new DeadhandTriggerRegistry(this.kv);
  }

  // --------------------------------------------------------------------------
  // Triggers (`cron` / `webhook` / `cas-diff` / `git`)
  // --------------------------------------------------------------------------

  public registerTrigger(def: DeadhandTriggerDef): void {
    this.registry.addTrigger(def);
  }

  public removeTrigger(id: string): boolean {
    return this.registry.removeTrigger(id);
  }

  public listTriggers(): readonly DeadhandTriggerDef[] {
    return this.registry.listTriggers();
  }

  public async fireTrigger<T = unknown>(
    triggerId: string,
    payload: unknown,
    executor: (ctx: Zer0qExecutionContext) => Promise<Zer0qExecutorOutput<T> | T> | Zer0qExecutorOutput<T> | T,
  ): Promise<DeadhandFireResult<T>> {
    return this.registry.fire(triggerId, payload, this.runner, executor);
  }

  public async fireCron<T = unknown>(
    triggerId: string,
    tickIso: string,
    executor: (ctx: Zer0qExecutionContext) => Promise<Zer0qExecutorOutput<T> | T> | Zer0qExecutorOutput<T> | T,
  ): Promise<DeadhandFireResult<T>> {
    const def = this.registry.getTrigger(triggerId);
    if (!def || def.kind !== 'cron') {
      throw new Error(`Trigger "${triggerId}" is not a registered cron trigger`);
    }
    return this.registry.fire(triggerId, { kind: 'cron', tick: tickIso }, this.runner, executor);
  }

  public async fireWebhook<T = unknown>(
    triggerId: string,
    deliveryId: string,
    body: unknown,
    executor: (ctx: Zer0qExecutionContext) => Promise<Zer0qExecutorOutput<T> | T> | Zer0qExecutorOutput<T> | T,
  ): Promise<DeadhandFireResult<T>> {
    const def = this.registry.getTrigger(triggerId);
    if (!def || def.kind !== 'webhook') {
      throw new Error(`Trigger "${triggerId}" is not a registered webhook trigger`);
    }
    return this.registry.fire(
      triggerId,
      { kind: 'webhook', deliveryId, body },
      this.runner,
      executor,
    );
  }

  public async fireCasDiff<T = unknown>(
    triggerId: string,
    diff: CasDiffPayload,
    executor: (ctx: Zer0qExecutionContext) => Promise<Zer0qExecutorOutput<T> | T> | Zer0qExecutorOutput<T> | T,
  ): Promise<DeadhandCasDiffResult<T>> {
    const def = this.registry.getTrigger(triggerId);
    if (!def || def.kind !== 'cas-diff') {
      throw new Error(`Trigger "${triggerId}" is not a registered cas-diff trigger`);
    }
    if (diff.prevTreeDigest === diff.nextTreeDigest) {
      return Object.freeze({ skippedUnchanged: true });
    }
    const fireResult = await this.registry.fire(
      triggerId,
      {
        kind: 'cas-diff',
        prevTreeDigest: diff.prevTreeDigest,
        nextTreeDigest: diff.nextTreeDigest,
        changedFiles: diff.changedFiles ?? [],
      },
      this.runner,
      executor,
    );
    return Object.freeze({
      skippedUnchanged: false,
      fireResult,
    });
  }

  // --------------------------------------------------------------------------
  // Atomic Alias Management (`set` / `get` / `rollback`)
  // --------------------------------------------------------------------------

  public setAlias(name: string, casHash: string): AliasMutationResult {
    const prev = this.kv.getAlias(name);
    const updated = this.kv.compareAndSwap(name, prev ?? null, casHash);
    if (!updated) {
      return Object.freeze({ ok: false, name, previousHash: prev, currentHash: this.kv.getAlias(name) });
    }
    if (prev !== undefined && prev !== casHash) {
      const stack = this.aliasHistory.get(name) ?? [];
      stack.push(prev);
      this.aliasHistory.set(name, stack);
    }
    return Object.freeze({
      ok: true,
      name,
      previousHash: prev,
      currentHash: casHash,
    });
  }

  public getAlias(name: string): string | undefined {
    return this.kv.getAlias(name);
  }

  public listAliases(): Readonly<Record<string, string>> {
    return this.kv.listAliases();
  }

  public getAliasHistory(name: string): readonly string[] {
    return Object.freeze([...(this.aliasHistory.get(name) ?? [])]);
  }

  public rollbackAlias(name: string): AliasMutationResult {
    const current = this.kv.getAlias(name);
    const stack = this.aliasHistory.get(name);
    if (!stack || stack.length === 0) {
      return Object.freeze({
        ok: false,
        name,
        previousHash: current,
        currentHash: current,
        rolledBack: false,
      });
    }
    const target = stack.pop()!;
    const swapped = this.kv.compareAndSwap(name, current ?? null, target);
    if (!swapped) {
      stack.push(target);
      return Object.freeze({
        ok: false,
        name,
        previousHash: current,
        currentHash: this.kv.getAlias(name),
        rolledBack: false,
      });
    }
    return Object.freeze({
      ok: true,
      name,
      previousHash: current,
      currentHash: target,
      rolledBack: true,
    });
  }

  // --------------------------------------------------------------------------
  // TTL Lease Heartbeat + Orphan Reaper
  // --------------------------------------------------------------------------

  public acquireLease(
    runId: string,
    opts: { readonly owner: string; readonly sandboxId?: string; readonly ttlMs: number; readonly nowMs?: number },
  ): LeaseRecord | null {
    return this.kv.acquireLease(runId, opts);
  }

  public heartbeatLease(
    runId: string,
    owner: string,
    ttlMs?: number,
    nowMs?: number,
  ): LeaseRecord | null {
    return this.kv.renewLease(runId, owner, ttlMs, nowMs);
  }

  public releaseLease(runId: string, owner: string): boolean {
    return this.kv.releaseLease(runId, owner);
  }

  public listActiveLeases(nowMs: number = Date.now()): readonly LeaseRecord[] {
    const out: LeaseRecord[] = [];
    for (const runId of this.kv.listLeaseRunIds()) {
      const lease = this.kv.getLease(runId, nowMs);
      if (lease) {
        out.push(lease);
      }
    }
    return Object.freeze(out);
  }

  public reapOrphanLeases(nowMs: number = Date.now()): readonly LeaseRecord[] {
    return this.kv.reapExpiredLeases(nowMs);
  }

  public startOrphanReaper(intervalMs = 5000, onReaped?: (reaped: readonly LeaseRecord[]) => void): void {
    this.stopOrphanReaper();
    const timer = setInterval(() => {
      const reaped = this.reapOrphanLeases();
      if (reaped.length > 0 && onReaped) {
        onReaped(reaped);
      }
    }, Math.max(10, intervalMs));
    timer.unref();
    this.reaperTimer = timer;
  }

  public stopOrphanReaper(): void {
    if (this.reaperTimer) {
      clearInterval(this.reaperTimer);
      this.reaperTimer = null;
    }
  }

  // --------------------------------------------------------------------------
  // Deterministic Replay
  // --------------------------------------------------------------------------

  public async replay<T = unknown>(
    runIdOrDigest: string,
    overrideModel?: string,
    executor?: (hydrated: Zer0qReplaySnapshot<T>) => Promise<T> | T,
  ): Promise<Zer0qReplayResult<T>> {
    return this.runner.replay<T>(runIdOrDigest, {
      ...(overrideModel !== undefined ? { modelOverride: overrideModel } : {}),
      ...(executor !== undefined ? { executor } : {}),
    });
  }

  // --------------------------------------------------------------------------
  // State Persistence Helper (for CLI / Daemon restarts)
  // --------------------------------------------------------------------------

  public exportState(): PersistedDeadhandState {
    const historyOut: Record<string, readonly string[]> = {};
    for (const [k, stack] of this.aliasHistory.entries()) {
      historyOut[k] = [...stack];
    }
    return Object.freeze({
      aliases: this.kv.listAliases(),
      aliasHistory: Object.freeze(historyOut),
      leases: this.kv.listAllLeases(),
      triggers: this.registry.listTriggers(),
    });
  }

  public importState(state: PersistedDeadhandState): void {
    for (const [k, v] of Object.entries(state.aliases ?? {})) {
      this.kv.setAlias(k, v);
    }
    for (const [k, stack] of Object.entries(state.aliasHistory ?? {})) {
      this.aliasHistory.set(k, [...stack]);
    }
    for (const lease of state.leases ?? []) {
      this.kv.restoreLease(lease);
    }
    for (const trig of state.triggers ?? []) {
      this.registry.addTrigger(trig);
    }
  }

  public saveStateFile(filePath: string): void {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, `${JSON.stringify(this.exportState(), null, 2)}\n`, 'utf8');
  }

  public loadStateFile(filePath: string): boolean {
    if (!fs.existsSync(filePath)) {
      return false;
    }
    const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8')) as PersistedDeadhandState;
    this.importState(parsed);
    return true;
  }
}

export function buildIdempotencyKey(triggerId: string, payload: unknown): string {
  return `idem:${triggerId}:${computeInputDigest(payload)}`;
}

export type { Zer0qJobManifest };
