/**
 * @file packages/zer0q-substrate/index.mts
 * @description ZER0Q Autonomous Agent Runner & SUBSTRATE CAS/KV Control Plane.
 *
 * Implements:
 *   - `canonicalJson` + `computeInputDigest` (SHA-512)
 *   - `SubstrateCasStore` (`v11/files/<aa>/<sha512>` put/get/has/verify/pin/gc)
 *   - `SubstrateKvStore` (mutable aliases + atomic `compareAndSwap`, TTL leases, idempotency dedup, token & wall-time budgets)
 *   - `Zer0qRunner` (`SKIMMER` `< 2 ms` / `0`-token CAS short-circuit + deterministic `replay()`)
 *   - `DeadhandTriggerRegistry` (cron/webhook/git/cas-diff triggers with KV idempotency dedup)
 *
 * Strictly `erasableSyntaxOnly: true` and Node 22+ built-ins only.
 */

import { createHash, randomBytes } from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { performance } from 'node:perf_hooks';
import { redactValue } from './secret-redactor.mts';

// ============================================================================
// 1. Canonical JSON & SHA-512 Digests
// ============================================================================

/**
 * Serializes any JSON-compatible value into deterministic canonical JSON
 * with lexicographically sorted object keys and `undefined` properties omitted.
 */
export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => (item === undefined ? 'null' : canonicalJson(item))).join(',')}]`;
  }
  const record = value as Record<string, unknown>;
  const keys = Object.keys(record)
    .filter((k) => record[k] !== undefined)
    .sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalJson(record[k])}`).join(',')}}`;
}

/**
 * Computes lowercase hexadecimal SHA-512 digest over UTF-8 text or raw bytes.
 */
export function sha512Hex(data: Uint8Array | string): string {
  return createHash('sha512').update(data).digest('hex');
}

/**
 * Computes deterministic SHA-512 digest over canonicalized job inputs.
 */
export function computeInputDigest(input: unknown): string {
  if (input instanceof Uint8Array) {
    return sha512Hex(input);
  }
  return sha512Hex(canonicalJson(input));
}

// ============================================================================
// 2. SubstrateCasStore (v11/files/<aa>/<sha512>)
// ============================================================================

export interface CasPutOptions {
  readonly executable?: boolean;
  readonly meta?: Readonly<Record<string, unknown>>;
}

export interface CasPutResult {
  readonly sha512: string;
  readonly path: string;
  readonly size: number;
  readonly executable: boolean;
  readonly deduplicated: boolean;
}

const SHA512_HEX_RE = /^[0-9a-f]{128}$/;

function normalizeDigest(sha512: string): string {
  const norm = sha512.trim().toLowerCase();
  if (!SHA512_HEX_RE.test(norm)) {
    throw new Error(`Invalid SHA-512 digest: "${sha512}"`);
  }
  return norm;
}

function resolveV11Root(baseRoot: string): string {
  const resolved = path.resolve(baseRoot).replace(/[\\/]+$/, '');
  return path.basename(resolved) === 'v11' ? resolved : path.join(resolved, 'v11');
}

/**
 * Content-addressed immutable store using `v11/files/<aa>/<sha512>[-exec]` layout.
 */
export class SubstrateCasStore {
  public readonly storeRoot: string;
  public readonly filesDir: string;
  public readonly tmpDir: string;
  private readonly pinsPath: string;
  private readonly pins: Map<string, Set<string>>;

  constructor(storeRoot: string) {
    this.storeRoot = resolveV11Root(storeRoot);
    this.filesDir = path.join(this.storeRoot, 'files');
    this.tmpDir = path.join(this.storeRoot, 'tmp');
    this.pinsPath = path.join(this.storeRoot, 'pins.json');
    this.pins = new Map<string, Set<string>>();
    this.ensureLayout();
    this.loadPins();
  }

  private ensureLayout(): void {
    fs.mkdirSync(this.filesDir, { recursive: true });
    fs.mkdirSync(this.tmpDir, { recursive: true });
  }

  private loadPins(): void {
    if (!fs.existsSync(this.pinsPath)) {
      return;
    }
    try {
      const parsed = JSON.parse(fs.readFileSync(this.pinsPath, 'utf8')) as Record<string, string[]>;
      for (const [digest, tags] of Object.entries(parsed)) {
        if (Array.isArray(tags) && tags.length > 0) {
          this.pins.set(digest, new Set(tags));
        }
      }
    } catch {
      // Ignore corrupt pin metadata on recovery
    }
  }

  private persistPins(): void {
    const out: Record<string, string[]> = {};
    for (const [digest, tags] of this.pins.entries()) {
      if (tags.size > 0) {
        out[digest] = Array.from(tags).sort();
      }
    }
    fs.writeFileSync(this.pinsPath, `${JSON.stringify(out, null, 2)}\n`, 'utf8');
  }

  /**
   * Computes exact `v11/files/<aa>/<sha512>[-exec]` disk path.
   */
  public blobPath(sha512: string, executable = false): string {
    const digest = normalizeDigest(sha512);
    const shard = digest.slice(0, 2);
    const filename = executable ? `${digest}-exec` : digest;
    return path.join(this.filesDir, shard, filename);
  }

  /**
   * Stores raw bytes, UTF-8 string, or canonical JSON value into CAS.
   */
  public put(
    data: Uint8Array | string | Readonly<Record<string, unknown>> | readonly unknown[],
    opts?: CasPutOptions,
  ): CasPutResult {
    this.ensureLayout();
    const bytes =
      data instanceof Uint8Array
        ? data
        : typeof data === 'string'
          ? Buffer.from(data, 'utf8')
          : Buffer.from(canonicalJson(data), 'utf8');

    const executable = opts?.executable === true;
    const digest = sha512Hex(bytes);
    const target = this.blobPath(digest, executable);
    fs.mkdirSync(path.dirname(target), { recursive: true });

    let deduplicated = false;
    if (fs.existsSync(target)) {
      deduplicated = true;
    } else {
      const tmp = path.join(this.tmpDir, `${process.pid}-${randomBytes(6).toString('hex')}`);
      fs.writeFileSync(tmp, bytes);
      if (executable && process.platform !== 'win32') {
        fs.chmodSync(tmp, 0o755);
      }
      try {
        fs.renameSync(tmp, target);
      } catch (err) {
        if (fs.existsSync(target)) {
          fs.rmSync(tmp, { force: true });
          deduplicated = true;
        } else {
          fs.rmSync(tmp, { force: true });
          throw err;
        }
      }
    }

    return Object.freeze({
      sha512: digest,
      path: target,
      size: bytes.byteLength,
      executable,
      deduplicated,
    });
  }

  public has(sha512: string, executable = false): boolean {
    return fs.existsSync(this.blobPath(sha512, executable));
  }

  public get(sha512: string, executable = false): Uint8Array | null {
    const target = this.blobPath(sha512, executable);
    if (!fs.existsSync(target)) {
      return null;
    }
    return fs.readFileSync(target);
  }

  public getJson<T = unknown>(sha512: string): T | null {
    const raw = this.get(sha512, false);
    if (!raw) {
      return null;
    }
    return JSON.parse(Buffer.from(raw).toString('utf8')) as T;
  }

  /**
   * Re-computes SHA-512 over stored bytes to verify cryptographic integrity.
   */
  public verify(sha512: string, executable = false): boolean {
    const digest = normalizeDigest(sha512);
    const bytes = this.get(digest, executable);
    if (bytes === null) {
      return false;
    }
    return sha512Hex(bytes) === digest;
  }

  public pin(sha512: string, tag = 'default'): void {
    const digest = normalizeDigest(sha512);
    let set = this.pins.get(digest);
    if (!set) {
      set = new Set<string>();
      this.pins.set(digest, set);
    }
    set.add(tag);
    this.persistPins();
  }

  public unpin(sha512: string, tag?: string): boolean {
    const digest = normalizeDigest(sha512);
    const set = this.pins.get(digest);
    if (!set) {
      return false;
    }
    if (tag === undefined) {
      this.pins.delete(digest);
    } else {
      set.delete(tag);
      if (set.size === 0) {
        this.pins.delete(digest);
      }
    }
    this.persistPins();
    return true;
  }

  public isPinned(sha512: string): boolean {
    const digest = normalizeDigest(sha512);
    const set = this.pins.get(digest);
    return set !== undefined && set.size > 0;
  }

  public listPins(): readonly string[] {
    return Object.freeze(Array.from(this.pins.keys()).sort());
  }

  /**
   * Reclaims all unpinned blobs from `v11/files/**`.
   */
  public gc(): { readonly removed: number; readonly retained: number } {
    let removed = 0;
    let retained = 0;
    if (!fs.existsSync(this.filesDir)) {
      return { removed, retained };
    }
    for (const shard of fs.readdirSync(this.filesDir)) {
      const shardDir = path.join(this.filesDir, shard);
      if (!fs.statSync(shardDir).isDirectory()) {
        continue;
      }
      for (const file of fs.readdirSync(shardDir)) {
        const digest = file.endsWith('-exec') ? file.slice(0, -5) : file;
        if (this.pins.has(digest)) {
          retained += 1;
        } else {
          fs.rmSync(path.join(shardDir, file), { force: true });
          removed += 1;
        }
      }
    }
    return { removed, retained };
  }
}

// ============================================================================
// 3. SubstrateKvStore (Aliases, Atomic CAS, Leases, Idempotency, Budgets)
// ============================================================================

export interface LeaseRecord {
  readonly runId: string;
  readonly owner: string;
  readonly sandboxId: string;
  readonly startedAt: number;
  readonly heartbeatAt: number;
  readonly expiresAt: number;
  readonly ttlMs: number;
}

export interface BudgetLimits {
  readonly maxTokens: number;
  readonly maxWallMs: number;
  readonly maxToolCalls?: number;
}

export interface BudgetState {
  readonly runId: string;
  readonly maxTokens: number;
  readonly tokensUsed: number;
  readonly maxWallMs: number;
  readonly elapsedMs: number;
  readonly maxToolCalls: number;
  readonly toolCallsUsed: number;
  readonly startedAt: number;
}

export interface BudgetCheckResult {
  readonly ok: boolean;
  readonly exceededReason?: 'TOKEN_CEILING_EXCEEDED' | 'WALL_TIME_EXCEEDED' | 'TOOL_CALL_CEILING_EXCEEDED';
  readonly remainingTokens: number;
  readonly remainingWallMs: number;
  readonly state: BudgetState;
}

interface IdempotencyEntry {
  readonly key: string;
  readonly value: string;
  readonly expiresAt: number | null;
}

/**
 * Sub-millisecond control-plane KV store for mutable aliases, atomic compare-and-swap,
 * TTL leases, trigger idempotency deduplication, and live token/wall-time budgets.
 */
export class SubstrateKvStore {
  private readonly aliases = new Map<string, string>();
  private readonly leases = new Map<string, LeaseRecord>();
  private readonly idempotency = new Map<string, IdempotencyEntry>();
  private readonly budgets = new Map<string, BudgetState>();

  public setAlias(name: string, casHash: string): void {
    this.aliases.set(name, casHash);
  }

  public getAlias(name: string): string | undefined {
    return this.aliases.get(name);
  }

  public deleteAlias(name: string): boolean {
    return this.aliases.delete(name);
  }

  public listAliases(): Readonly<Record<string, string>> {
    const out: Record<string, string> = {};
    for (const [k, v] of this.aliases.entries()) {
      out[k] = v;
    }
    return Object.freeze(out);
  }

  /**
   * Atomic compare-and-swap on an alias key.
   * Pass `expected === null` (or `undefined`) to perform a `SETNX` create-if-absent.
   */
  public compareAndSwap(
    key: string,
    expected: string | null | undefined,
    nextValue: string,
  ): boolean {
    const current = this.aliases.get(key);
    if (expected === null || expected === undefined) {
      if (current !== undefined) {
        return false;
      }
      this.aliases.set(key, nextValue);
      return true;
    }
    if (current !== expected) {
      return false;
    }
    this.aliases.set(key, nextValue);
    return true;
  }

  public acquireLease(
    runId: string,
    opts: { readonly owner: string; readonly sandboxId?: string; readonly ttlMs: number; readonly nowMs?: number },
  ): LeaseRecord | null {
    const now = opts.nowMs ?? Date.now();
    const existing = this.leases.get(runId);
    if (existing && existing.expiresAt > now) {
      return null;
    }
    const ttlMs = Math.max(1, opts.ttlMs);
    const lease: LeaseRecord = Object.freeze({
      runId,
      owner: opts.owner,
      sandboxId: opts.sandboxId ?? `sbx-${runId}`,
      startedAt: now,
      heartbeatAt: now,
      expiresAt: now + ttlMs,
      ttlMs,
    });
    this.leases.set(runId, lease);
    return lease;
  }

  public renewLease(
    runId: string,
    owner: string,
    ttlMs?: number,
    nowMs?: number,
  ): LeaseRecord | null {
    const now = nowMs ?? Date.now();
    const existing = this.leases.get(runId);
    if (!existing || existing.owner !== owner || existing.expiresAt <= now) {
      return null;
    }
    const nextTtl = Math.max(1, ttlMs ?? existing.ttlMs);
    const renewed: LeaseRecord = Object.freeze({
      ...existing,
      heartbeatAt: now,
      expiresAt: now + nextTtl,
      ttlMs: nextTtl,
    });
    this.leases.set(runId, renewed);
    return renewed;
  }

  public releaseLease(runId: string, owner: string): boolean {
    const existing = this.leases.get(runId);
    if (!existing || existing.owner !== owner) {
      return false;
    }
    return this.leases.delete(runId);
  }

  public getLease(runId: string, nowMs?: number): LeaseRecord | null {
    const now = nowMs ?? Date.now();
    const existing = this.leases.get(runId);
    if (!existing || existing.expiresAt <= now) {
      return null;
    }
    return existing;
  }

  public listLeaseRunIds(): readonly string[] {
    return Object.freeze(Array.from(this.leases.keys()));
  }

  public listAllLeases(): readonly LeaseRecord[] {
    return Object.freeze(Array.from(this.leases.values()));
  }

  public restoreLease(lease: LeaseRecord): void {
    this.leases.set(lease.runId, Object.freeze({ ...lease }));
  }

  public reapExpiredLeases(nowMs: number = Date.now()): readonly LeaseRecord[] {
    const reaped: LeaseRecord[] = [];
    for (const [runId, lease] of this.leases.entries()) {
      if (lease.expiresAt <= nowMs) {
        this.leases.delete(runId);
        reaped.push(lease);
      }
    }
    return Object.freeze(reaped);
  }

  public claimIdempotencyKey(
    key: string,
    value: string,
    ttlMs?: number,
    nowMs: number = Date.now(),
  ): { readonly acquired: boolean; readonly existing?: string } {
    const cur = this.idempotency.get(key);
    if (cur && (cur.expiresAt === null || cur.expiresAt > nowMs)) {
      return { acquired: false, existing: cur.value };
    }
    this.idempotency.set(key, {
      key,
      value,
      expiresAt: ttlMs !== undefined ? nowMs + Math.max(1, ttlMs) : null,
    });
    return { acquired: true };
  }

  public getIdempotencyKey(key: string, nowMs: number = Date.now()): string | undefined {
    const cur = this.idempotency.get(key);
    if (!cur || (cur.expiresAt !== null && cur.expiresAt <= nowMs)) {
      return undefined;
    }
    return cur.value;
  }

  public initBudget(runId: string, limits: BudgetLimits, nowMs: number = Date.now()): BudgetState {
    const state: BudgetState = Object.freeze({
      runId,
      maxTokens: Math.max(0, limits.maxTokens),
      tokensUsed: 0,
      maxWallMs: Math.max(0, limits.maxWallMs),
      elapsedMs: 0,
      maxToolCalls: limits.maxToolCalls ?? Number.MAX_SAFE_INTEGER,
      toolCallsUsed: 0,
      startedAt: nowMs,
    });
    this.budgets.set(runId, state);
    return state;
  }

  public recordBudgetTick(
    runId: string,
    delta: { readonly tokensUsed?: number; readonly toolCalls?: number; readonly elapsedMs?: number },
    nowMs: number = Date.now(),
  ): BudgetCheckResult {
    const prev = this.budgets.get(runId);
    if (!prev) {
      throw new Error(`No budget initialized for run "${runId}"`);
    }
    const tokensUsed = prev.tokensUsed + (delta.tokensUsed ?? 0);
    const toolCallsUsed = prev.toolCallsUsed + (delta.toolCalls ?? 0);
    const elapsedMs = delta.elapsedMs ?? Math.max(0, nowMs - prev.startedAt);
    const nextState: BudgetState = Object.freeze({
      ...prev,
      tokensUsed,
      toolCallsUsed,
      elapsedMs,
    });
    this.budgets.set(runId, nextState);

    let exceededReason: BudgetCheckResult['exceededReason'];
    if (tokensUsed > nextState.maxTokens) {
      exceededReason = 'TOKEN_CEILING_EXCEEDED';
    } else if (elapsedMs > nextState.maxWallMs) {
      exceededReason = 'WALL_TIME_EXCEEDED';
    } else if (toolCallsUsed > nextState.maxToolCalls) {
      exceededReason = 'TOOL_CALL_CEILING_EXCEEDED';
    }

    return Object.freeze({
      ok: exceededReason === undefined,
      ...(exceededReason ? { exceededReason } : {}),
      remainingTokens: Math.max(0, nextState.maxTokens - tokensUsed),
      remainingWallMs: Math.max(0, nextState.maxWallMs - elapsedMs),
      state: nextState,
    });
  }

  public getBudget(runId: string): BudgetState | undefined {
    return this.budgets.get(runId);
  }
}

// ============================================================================
// 4. Zer0qRunner (SKIMMER < 2 ms Short-Circuit + Replay)
// ============================================================================

export type Zer0qCachePolicy = 'zero-q-lookup-or-run' | 'bypass';

export interface Zer0qJobInputs {
  readonly agentsMd: string;
  readonly skills: readonly string[];
  readonly workspace: string;
  readonly prompt: string;
  readonly toolchainHash?: string;
}

export interface Zer0qExecutionConfig {
  readonly model: string;
  readonly cachePolicy?: Zer0qCachePolicy;
  readonly maxTokens: number;
  readonly maxWallMs: number;
  readonly maxToolCalls?: number;
  readonly leaseTtlMs?: number;
}

export interface Zer0qJobManifest {
  readonly apiVersion: 'zer0q.awesome-os/v1';
  readonly name: string;
  readonly alias?: string;
  readonly idempotencyKey?: string;
  readonly inputs: Zer0qJobInputs;
  readonly execution: Zer0qExecutionConfig;
}

export interface SkimmerLookupResult<T = unknown> {
  readonly hit: boolean;
  readonly inputDigest: string;
  readonly outputDigest?: string;
  readonly runId?: string;
  readonly output?: T;
  readonly latencyMs: number;
  readonly tokensUsed: 0;
}

export interface Zer0qCommitRecord<T = unknown> {
  readonly runId: string;
  readonly inputDigest: string;
  readonly manifest: Zer0qJobManifest;
  readonly output: T;
  readonly tokensUsed: number;
  readonly committedAt: number;
}

export type Zer0qRunStatus = 'CACHE_HIT' | 'COMMITTED' | 'DEDUPLICATED';

export interface Zer0qRunResult<T = unknown> {
  readonly status: Zer0qRunStatus;
  readonly cached: boolean;
  readonly runId: string;
  readonly inputDigest: string;
  readonly outputDigest: string;
  readonly output: T;
  readonly tokensUsed: number;
  readonly latencyMs: number;
}

export interface Zer0qExecutionContext {
  readonly runId: string;
  readonly inputDigest: string;
  readonly manifest: Zer0qJobManifest;
  readonly recordTokens: (tokens: number, toolCalls?: number) => BudgetCheckResult;
}

export interface Zer0qReplaySnapshot<T = unknown> {
  readonly runId: string;
  readonly inputDigest: string;
  readonly originalDigest: string;
  readonly manifest: Zer0qJobManifest;
  readonly originalOutput: T;
  readonly verified: boolean;
}

export interface Zer0qReplayResult<T = unknown> {
  readonly hydrated: Zer0qReplaySnapshot<T>;
  readonly reexecuted: boolean;
  readonly modelUsed: string;
  readonly replayOutput: T;
  readonly outputDigest: string;
  readonly latencyMs: number;
}

export interface Zer0qExecutorOutput<T = unknown> {
  readonly output: T;
  readonly tokensUsed?: number;
}

export class Zer0qRunner {
  public readonly cas: SubstrateCasStore;
  public readonly kv: SubstrateKvStore;
  private readonly skimmerCache = new Map<
    string,
    { readonly outputDigest: string; readonly record: Zer0qCommitRecord<unknown> }
  >();
  private runSeq = 0;

  constructor(cas: SubstrateCasStore, kv: SubstrateKvStore = new SubstrateKvStore()) {
    this.cas = cas;
    this.kv = kv;
  }

  /**
   * Computes canonical SKIMMER digest over instructions, skills, workspace, prompt, and model.
   */
  public computeManifestDigest(manifest: Zer0qJobManifest): string {
    return computeInputDigest({
      apiVersion: manifest.apiVersion,
      inputs: manifest.inputs,
      model: manifest.execution.model,
    });
  }

  /**
   * SKIMMER pre-flight resolver: checks KV->CAS index in `< 2 ms` with `0` tokens.
   */
  public lookup<T = unknown>(manifest: Zer0qJobManifest): SkimmerLookupResult<T> {
    const t0 = performance.now();
    const inputDigest = this.computeManifestDigest(manifest);
    const commitDigest = this.kv.getAlias(`skimmer:${inputDigest}`);
    if (commitDigest) {
      const cached = this.skimmerCache.get(inputDigest);
      if (cached && cached.outputDigest === commitDigest) {
        const rec = cached.record as Zer0qCommitRecord<T>;
        return Object.freeze({
          hit: true,
          inputDigest,
          outputDigest: commitDigest,
          runId: rec.runId,
          output: rec.output,
          latencyMs: Math.max(0, performance.now() - t0),
          tokensUsed: 0,
        });
      }
      const bytes = this.cas.get(commitDigest, false);
      if (bytes !== null && sha512Hex(bytes) === commitDigest.toLowerCase()) {
        const record = JSON.parse(Buffer.from(bytes).toString('utf8')) as Zer0qCommitRecord<T>;
        if (record && record.inputDigest === inputDigest) {
          this.skimmerCache.set(inputDigest, { outputDigest: commitDigest, record });
          return Object.freeze({
            hit: true,
            inputDigest,
            outputDigest: commitDigest,
            runId: record.runId,
            output: record.output,
            latencyMs: Math.max(0, performance.now() - t0),
            tokensUsed: 0,
          });
        }
      }
    }
    return Object.freeze({
      hit: false,
      inputDigest,
      latencyMs: Math.max(0, performance.now() - t0),
      tokensUsed: 0,
    });
  }

  /**
   * Executes a declarative job manifest with SKIMMER CAS short-circuit, optimistic lease,
   * budget tracking, and CAS commit.
   */
  public async run<T = unknown>(
    manifest: Zer0qJobManifest,
    executor: (ctx: Zer0qExecutionContext) => Promise<Zer0qExecutorOutput<T> | T> | Zer0qExecutorOutput<T> | T,
  ): Promise<Zer0qRunResult<T>> {
    const t0 = performance.now();
    const policy = manifest.execution.cachePolicy ?? 'zero-q-lookup-or-run';
    if (policy === 'zero-q-lookup-or-run') {
      const hit = this.lookup<T>(manifest);
      if (hit.hit && hit.outputDigest !== undefined && hit.runId !== undefined) {
        return Object.freeze({
          status: 'CACHE_HIT',
          cached: true,
          runId: hit.runId,
          inputDigest: hit.inputDigest,
          outputDigest: hit.outputDigest,
          output: hit.output as T,
          tokensUsed: 0,
          latencyMs: hit.latencyMs,
        });
      }
    }

    const inputDigest = this.computeManifestDigest(manifest);

    if (manifest.idempotencyKey) {
      const existingDigest = this.kv.getIdempotencyKey(manifest.idempotencyKey);
      if (existingDigest && this.cas.verify(existingDigest)) {
        const prev = this.cas.getJson<Zer0qCommitRecord<T>>(existingDigest);
        if (prev) {
          return Object.freeze({
            status: 'DEDUPLICATED',
            cached: true,
            runId: prev.runId,
            inputDigest: prev.inputDigest,
            outputDigest: existingDigest,
            output: prev.output,
            tokensUsed: 0,
            latencyMs: Math.max(0, performance.now() - t0),
          });
        }
      }
    }

    this.runSeq += 1;
    const runId = `run-${this.runSeq}-${inputDigest.slice(0, 12)}`;
    const owner = `zer0q-runner:${process.pid}`;
    const lease = this.kv.acquireLease(runId, {
      owner,
      ttlMs: manifest.execution.leaseTtlMs ?? 60_000,
    });
    if (!lease) {
      throw new Error(`Failed to acquire lease for run "${runId}"`);
    }

    this.kv.initBudget(runId, {
      maxTokens: manifest.execution.maxTokens,
      maxWallMs: manifest.execution.maxWallMs,
      maxToolCalls: manifest.execution.maxToolCalls,
    });

    let trackedTokens = 0;
    const ctx: Zer0qExecutionContext = {
      runId,
      inputDigest,
      manifest,
      recordTokens: (tokens: number, toolCalls = 1): BudgetCheckResult => {
        trackedTokens += tokens;
        const check = this.kv.recordBudgetTick(runId, { tokensUsed: tokens, toolCalls });
        if (!check.ok) {
          throw new Error(`Budget exceeded for ${runId}: ${check.exceededReason}`);
        }
        return check;
      },
    };

    try {
      const rawRes = await executor(ctx);
      const isEnvelope =
        rawRes !== null &&
        typeof rawRes === 'object' &&
        'output' in rawRes &&
        Object.keys(rawRes as Record<string, unknown>).every(
          (k) => k === 'output' || k === 'tokensUsed',
        );
      const rawOutput = isEnvelope ? (rawRes as Zer0qExecutorOutput<T>).output : (rawRes as T);
      const tokensUsed = isEnvelope
        ? ((rawRes as Zer0qExecutorOutput<T>).tokensUsed ?? trackedTokens)
        : trackedTokens;

      const rawCommitRecord: Zer0qCommitRecord<T> = {
        runId,
        inputDigest,
        manifest,
        output: rawOutput,
        tokensUsed,
        committedAt: Date.now(),
      };
      const commitRecord = redactValue(rawCommitRecord).redacted;
      const output = commitRecord.output;
      const putRes = this.cas.put(commitRecord);
      this.cas.pin(putRes.sha512, runId);
      this.skimmerCache.set(inputDigest, { outputDigest: putRes.sha512, record: commitRecord });
      this.kv.setAlias(`skimmer:${inputDigest}`, putRes.sha512);
      this.kv.setAlias(`run:${runId}`, putRes.sha512);
      if (manifest.alias) {
        this.kv.setAlias(manifest.alias, putRes.sha512);
      }
      if (manifest.idempotencyKey) {
        this.kv.claimIdempotencyKey(manifest.idempotencyKey, putRes.sha512);
      }

      return Object.freeze({
        status: 'COMMITTED',
        cached: false,
        runId,
        inputDigest,
        outputDigest: putRes.sha512,
        output,
        tokensUsed,
        latencyMs: Math.max(0, performance.now() - t0),
      });
    } finally {
      this.kv.releaseLease(runId, owner);
    }
  }

  /**
   * Re-hydrates exact historical inputs & output from CAS by `runId` or SHA-512 digest,
   * verifies CAS integrity, and optionally re-executes against a newer model.
   */
  public async replay<T = unknown>(
    runIdOrDigest: string,
    opts?: {
      readonly modelOverride?: string;
      readonly executor?: (hydrated: Zer0qReplaySnapshot<T>) => Promise<T> | T;
    },
  ): Promise<Zer0qReplayResult<T>> {
    const t0 = performance.now();
    const digest = SHA512_HEX_RE.test(runIdOrDigest)
      ? runIdOrDigest
      : this.kv.getAlias(`run:${runIdOrDigest}`) ?? this.kv.getAlias(runIdOrDigest);
    if (!digest) {
      throw new Error(`Cannot resolve replay target "${runIdOrDigest}"`);
    }
    const verified = this.cas.verify(digest);
    if (!verified) {
      throw new Error(`CAS verification failed for replay digest "${digest}"`);
    }
    const record = this.cas.getJson<Zer0qCommitRecord<T>>(digest);
    if (!record) {
      throw new Error(`Missing CAS commit record for "${digest}"`);
    }

    const hydrated: Zer0qReplaySnapshot<T> = Object.freeze({
      runId: record.runId,
      inputDigest: record.inputDigest,
      originalDigest: digest,
      manifest: record.manifest,
      originalOutput: record.output,
      verified,
    });

    if (opts?.executor) {
      const rawReplayOutput = await opts.executor(hydrated);
      const replayOutput = redactValue(rawReplayOutput).redacted;
      const putRes = this.cas.put({
        replayedFrom: digest,
        model: opts.modelOverride ?? record.manifest.execution.model,
        output: replayOutput,
      });
      return Object.freeze({
        hydrated,
        reexecuted: true,
        modelUsed: opts.modelOverride ?? record.manifest.execution.model,
        replayOutput,
        outputDigest: putRes.sha512,
        latencyMs: Math.max(0, performance.now() - t0),
      });
    }

    return Object.freeze({
      hydrated,
      reexecuted: false,
      modelUsed: opts?.modelOverride ?? record.manifest.execution.model,
      replayOutput: record.output,
      outputDigest: digest,
      latencyMs: Math.max(0, performance.now() - t0),
    });
  }
}

// ============================================================================
// 5. DeadhandTriggerRegistry
// ============================================================================

export type DeadhandTriggerKind = 'cron' | 'webhook' | 'git' | 'cas-diff';

export interface DeadhandTriggerDef {
  readonly id: string;
  readonly kind: DeadhandTriggerKind;
  readonly scheduleOrRef: string;
  readonly manifest: Zer0qJobManifest;
  readonly idempotencyTtlMs?: number;
}

export interface DeadhandFireResult<T = unknown> {
  readonly triggerId: string;
  readonly idempotencyKey: string;
  readonly deduplicated: boolean;
  readonly runResult?: Zer0qRunResult<T>;
}

export class DeadhandTriggerRegistry {
  private readonly triggers = new Map<string, DeadhandTriggerDef>();
  private readonly kv: SubstrateKvStore;

  constructor(kv: SubstrateKvStore = new SubstrateKvStore()) {
    this.kv = kv;
  }

  public addTrigger(def: DeadhandTriggerDef): void {
    this.triggers.set(def.id, Object.freeze({ ...def }));
  }

  public removeTrigger(id: string): boolean {
    return this.triggers.delete(id);
  }

  public getTrigger(id: string): DeadhandTriggerDef | undefined {
    return this.triggers.get(id);
  }

  public listTriggers(): readonly DeadhandTriggerDef[] {
    return Object.freeze(Array.from(this.triggers.values()));
  }

  /**
   * Fires a registered trigger with KV idempotency deduplication so duplicate events
   * (`idem:<triggerId>:<eventDigest>`) never spawn twin runs.
   */
  public async fire<T = unknown>(
    triggerId: string,
    eventPayload: unknown,
    runner: Zer0qRunner,
    executor: (ctx: Zer0qExecutionContext) => Promise<Zer0qExecutorOutput<T> | T> | Zer0qExecutorOutput<T> | T,
  ): Promise<DeadhandFireResult<T>> {
    const def = this.triggers.get(triggerId);
    if (!def) {
      throw new Error(`Unknown Deadhand trigger "${triggerId}"`);
    }
    const eventDigest = computeInputDigest(eventPayload);
    const idempotencyKey = `idem:${triggerId}:${eventDigest}`;
    const claim = this.kv.claimIdempotencyKey(
      idempotencyKey,
      eventDigest,
      def.idempotencyTtlMs ?? 300_000,
    );
    if (!claim.acquired) {
      return Object.freeze({
        triggerId,
        idempotencyKey,
        deduplicated: true,
      });
    }

    const runResult = await runner.run(def.manifest, executor);
    return Object.freeze({
      triggerId,
      idempotencyKey,
      deduplicated: false,
      runResult,
    });
  }
}
