/**
 * @file packages/observability-eval/index.mts
 * @description NDJSON Flight Recorder (`p50`/`p95` latency stats), Deterministic `AgentEvalHarness`,
 * `SimulationRunner`, and `< 3 ms` Online Distillation `Crystallizer`.
 *
 * Strictly `erasableSyntaxOnly: true`, Node 22+ built-ins only.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { performance } from 'node:perf_hooks';

// ============================================================================
// 1. NDJSON Flight Recorder
// ============================================================================

export type FlightEventKind =
  | 'cas_hit'
  | 'tool_call'
  | 'moe_quarantine'
  | 'armor_block'
  | 'budget_tick'
  | 'eval_score';

export interface FlightEvent {
  readonly seq: number;
  readonly ts: number;
  readonly kind: FlightEventKind;
  readonly runId: string;
  readonly latencyMs: number;
  readonly tokensUsed?: number;
  readonly tokensSaved?: number;
  readonly detail?: string;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface RecordFlightEventInput {
  readonly kind: FlightEventKind;
  readonly runId: string;
  readonly latencyMs: number;
  readonly tokensUsed?: number;
  readonly tokensSaved?: number;
  readonly detail?: string;
  readonly metadata?: Readonly<Record<string, unknown>>;
  readonly ts?: number;
}

export interface LatencyStats {
  readonly count: number;
  readonly minMs: number;
  readonly maxMs: number;
  readonly avgMs: number;
  readonly p50Ms: number;
  readonly p95Ms: number;
}

export interface FlightSummary {
  readonly totalEvents: number;
  readonly countsByKind: Readonly<Record<FlightEventKind, number>>;
  readonly totalTokensUsed: number;
  readonly totalTokensSaved: number;
  readonly overallLatency: LatencyStats;
}

export function computeLatencyStats(latencies: readonly number[]): LatencyStats {
  if (latencies.length === 0) {
    return Object.freeze({ count: 0, minMs: 0, maxMs: 0, avgMs: 0, p50Ms: 0, p95Ms: 0 });
  }
  const sorted = [...latencies].sort((a, b) => a - b);
  const sum = sorted.reduce((acc, v) => acc + v, 0);
  const p50Idx = Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * 0.5));
  const p95Idx = Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * 0.95));
  return Object.freeze({
    count: sorted.length,
    minMs: sorted[0]!,
    maxMs: sorted[sorted.length - 1]!,
    avgMs: sum / sorted.length,
    p50Ms: sorted[p50Idx]!,
    p95Ms: sorted[p95Idx]!,
  });
}

export class FlightRecorder {
  private readonly events: FlightEvent[] = [];
  private seq = 0;

  public record(input: RecordFlightEventInput): FlightEvent {
    this.seq += 1;
    const ev: FlightEvent = Object.freeze({
      seq: this.seq,
      ts: input.ts ?? Date.now(),
      kind: input.kind,
      runId: input.runId,
      latencyMs: Math.max(0, input.latencyMs),
      ...(input.tokensUsed !== undefined ? { tokensUsed: input.tokensUsed } : {}),
      ...(input.tokensSaved !== undefined ? { tokensSaved: input.tokensSaved } : {}),
      ...(input.detail !== undefined ? { detail: input.detail } : {}),
      ...(input.metadata !== undefined ? { metadata: input.metadata } : {}),
    });
    this.events.push(ev);
    return ev;
  }

  public listEvents(kind?: FlightEventKind): readonly FlightEvent[] {
    if (!kind) {
      return Object.freeze([...this.events]);
    }
    return Object.freeze(this.events.filter((e) => e.kind === kind));
  }

  public getLatencyStats(kind?: FlightEventKind): LatencyStats {
    const target = kind ? this.events.filter((e) => e.kind === kind) : this.events;
    return computeLatencyStats(target.map((e) => e.latencyMs));
  }

  public getSummary(): FlightSummary {
    const countsByKind: Record<FlightEventKind, number> = {
      cas_hit: 0,
      tool_call: 0,
      moe_quarantine: 0,
      armor_block: 0,
      budget_tick: 0,
      eval_score: 0,
    };
    let totalTokensUsed = 0;
    let totalTokensSaved = 0;
    for (const ev of this.events) {
      countsByKind[ev.kind] += 1;
      totalTokensUsed += ev.tokensUsed ?? 0;
      totalTokensSaved += ev.tokensSaved ?? 0;
    }
    return Object.freeze({
      totalEvents: this.events.length,
      countsByKind: Object.freeze(countsByKind),
      totalTokensUsed,
      totalTokensSaved,
      overallLatency: this.getLatencyStats(),
    });
  }

  public toNdjson(): string {
    return this.events.map((e) => JSON.stringify(e)).join('\n') + (this.events.length > 0 ? '\n' : '');
  }

  public static fromNdjson(ndjson: string): readonly FlightEvent[] {
    return Object.freeze(
      ndjson
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0)
        .map((line) => JSON.parse(line) as FlightEvent),
    );
  }

  public writeToFile(filePath: string): void {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, this.toNdjson(), 'utf8');
  }

  public static readFromFile(filePath: string): readonly FlightEvent[] {
    if (!fs.existsSync(filePath)) return Object.freeze([]);
    return FlightRecorder.fromNdjson(fs.readFileSync(filePath, 'utf8'));
  }

  public createEventStream(): ReadableStream<string> {
    let idx = 0;
    const snapshot = [...this.events];
    return new ReadableStream<string>({
      pull(controller) {
        if (idx >= snapshot.length) {
          controller.close();
          return;
        }
        controller.enqueue(`${JSON.stringify(snapshot[idx])}\n`);
        idx += 1;
      },
    });
  }
}

// ============================================================================
// 2. Deterministic AgentEvalHarness & SimulationRunner
// ============================================================================

export interface EvalCase<TInput = unknown, TOutput = unknown> {
  readonly id: string;
  readonly input: TInput;
  readonly expected?: TOutput;
  readonly scoreFn?: (output: TOutput, expected?: TOutput) => number;
  readonly passThreshold?: number;
}

export interface EvalCaseResult<TOutput = unknown> {
  readonly id: string;
  readonly score: number;
  readonly passed: boolean;
  readonly latencyMs: number;
  readonly output?: TOutput;
  readonly error?: string;
}

export interface EvalReport<TOutput = unknown> {
  readonly suiteName: string;
  readonly totalCases: number;
  readonly passedCases: number;
  readonly passRate: number;
  readonly avgScore: number;
  readonly latencyStats: LatencyStats;
  readonly results: readonly EvalCaseResult<TOutput>[];
}

export class AgentEvalHarness {
  public readonly recorder: FlightRecorder;

  constructor(recorder: FlightRecorder = new FlightRecorder()) {
    this.recorder = recorder;
  }

  public async run<TInput, TOutput>(
    suiteName: string,
    cases: readonly EvalCase<TInput, TOutput>[],
    agentFn: (input: TInput) => Promise<TOutput> | TOutput,
  ): Promise<EvalReport<TOutput>> {
    const results: EvalCaseResult<TOutput>[] = [];
    const latencies: number[] = [];
    let passedCases = 0;
    let scoreSum = 0;

    for (const testCase of cases) {
      const t0 = performance.now();
      let score = 0;
      let passed = false;
      let output: TOutput | undefined;
      let errorMsg: string | undefined;

      try {
        output = await agentFn(testCase.input);
        if (testCase.scoreFn) {
          score = Math.max(0, Math.min(1, testCase.scoreFn(output, testCase.expected)));
        } else {
          score = JSON.stringify(output) === JSON.stringify(testCase.expected) ? 1 : 0;
        }
        passed = score >= (testCase.passThreshold ?? 1.0);
      } catch (err) {
        errorMsg = err instanceof Error ? err.message : String(err);
        score = 0;
        passed = false;
      }

      const latencyMs = Math.max(0, performance.now() - t0);
      latencies.push(latencyMs);
      scoreSum += score;
      if (passed) passedCases += 1;

      this.recorder.record({
        kind: 'eval_score',
        runId: `${suiteName}:${testCase.id}`,
        latencyMs,
        detail: passed ? 'PASS' : 'FAIL',
        metadata: { suiteName, caseId: testCase.id, score, passed, ...(errorMsg ? { error: errorMsg } : {}) },
      });

      results.push(
        Object.freeze({
          id: testCase.id,
          score,
          passed,
          latencyMs,
          ...(output !== undefined ? { output } : {}),
          ...(errorMsg !== undefined ? { error: errorMsg } : {}),
        }),
      );
    }

    const totalCases = cases.length;
    return Object.freeze({
      suiteName,
      totalCases,
      passedCases,
      passRate: totalCases > 0 ? passedCases / totalCases : 1,
      avgScore: totalCases > 0 ? scoreSum / totalCases : 1,
      latencyStats: computeLatencyStats(latencies),
      results: Object.freeze(results),
    });
  }
}

export interface SimulationStep<TState> {
  readonly name: string;
  readonly eventKind: FlightEventKind;
  readonly apply: (state: TState) => TState | Promise<TState>;
  readonly invariant?: (state: TState) => boolean;
}

export interface SimulationReport<TState> {
  readonly scenarioName: string;
  readonly stepsExecuted: number;
  readonly allInvariantsHeld: boolean;
  readonly failedSteps: readonly string[];
  readonly finalState: TState;
  readonly latencyStats: LatencyStats;
}

export class SimulationRunner {
  public readonly recorder: FlightRecorder;

  constructor(recorder: FlightRecorder = new FlightRecorder()) {
    this.recorder = recorder;
  }

  public async runScenario<TState>(
    scenarioName: string,
    initialState: TState,
    steps: readonly SimulationStep<TState>[],
  ): Promise<SimulationReport<TState>> {
    let state = initialState;
    const failedSteps: string[] = [];
    const latencies: number[] = [];

    for (const step of steps) {
      const t0 = performance.now();
      state = await step.apply(state);
      const latencyMs = Math.max(0, performance.now() - t0);
      latencies.push(latencyMs);

      const invariantOk = step.invariant ? step.invariant(state) : true;
      if (!invariantOk) {
        failedSteps.push(step.name);
      }

      this.recorder.record({
        kind: step.eventKind,
        runId: scenarioName,
        latencyMs,
        detail: `${step.name}:${invariantOk ? 'OK' : 'INVARIANT_FAILED'}`,
      });
    }

    return Object.freeze({
      scenarioName,
      stepsExecuted: steps.length,
      allInvariantsHeld: failedSteps.length === 0,
      failedSteps: Object.freeze(failedSteps),
      finalState: state,
      latencyStats: computeLatencyStats(latencies),
    });
  }
}

// ============================================================================
// 3. Crystallizer (3 ms Online Distillation Lookup Table)
// ============================================================================

export interface CrystallizedEntry<T = string> {
  readonly key: string;
  readonly normalizedInput: string;
  readonly solution: T;
  readonly sourceModel: string;
  readonly confidence: number;
  readonly trigrams: readonly string[];
  readonly distilledAt: number;
}

export interface CrystallizerLookupResult<T = string> {
  readonly hit: boolean;
  readonly similarity: number;
  readonly latencyMs: number;
  readonly entry?: CrystallizedEntry<T>;
}

function normalizePrompt(text: string): string {
  return text.toLowerCase().replace(/\s+/g, ' ').trim();
}

function trigramsSet(text: string): string[] {
  const norm = normalizePrompt(text);
  if (norm.length <= 3) return norm ? [norm] : [];
  const seen = new Set<string>();
  for (let i = 0; i <= norm.length - 3; i += 1) {
    seen.add(norm.slice(i, i + 3));
  }
  return Array.from(seen);
}

function jaccardSimilarity(a: readonly string[], b: readonly string[]): number {
  if (a.length === 0 || b.length === 0) return 0;
  const setB = new Set(b);
  let inter = 0;
  for (const item of a) {
    if (setB.has(item)) inter += 1;
  }
  const union = a.length + b.length - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * 3 ms online distillation lookup table that crystallizes teacher-LLM solutions
 * for zero-token deterministic reuse on exact or near-identical queries.
 */
export class Crystallizer<T = string> {
  private readonly exactIndex = new Map<string, CrystallizedEntry<T>>();
  private readonly entries: CrystallizedEntry<T>[] = [];
  private hits = 0;
  private misses = 0;
  private totalLookupMs = 0;

  public distill(
    input: string,
    solution: T,
    opts?: { readonly sourceModel?: string; readonly confidence?: number },
  ): CrystallizedEntry<T> {
    const normalizedInput = normalizePrompt(input);
    const entry: CrystallizedEntry<T> = Object.freeze({
      key: `crys-${this.entries.length + 1}`,
      normalizedInput,
      solution,
      sourceModel: opts?.sourceModel ?? 'teacher-llm',
      confidence: opts?.confidence ?? 1.0,
      trigrams: Object.freeze(trigramsSet(normalizedInput)),
      distilledAt: Date.now(),
    });
    const existing = this.exactIndex.get(normalizedInput);
    if (existing) {
      const idx = this.entries.indexOf(existing);
      if (idx >= 0) this.entries[idx] = entry;
    } else {
      this.entries.push(entry);
    }
    this.exactIndex.set(normalizedInput, entry);
    return entry;
  }

  public lookup(input: string, minSimilarity = 0.72): CrystallizerLookupResult<T> {
    const t0 = performance.now();
    const norm = normalizePrompt(input);
    const exact = this.exactIndex.get(norm);
    if (exact) {
      const latencyMs = Math.max(0, performance.now() - t0);
      this.hits += 1;
      this.totalLookupMs += latencyMs;
      return Object.freeze({ hit: true, similarity: 1.0, latencyMs, entry: exact });
    }

    const qGrams = trigramsSet(norm);
    let bestSim = 0;
    let bestEntry: CrystallizedEntry<T> | undefined;
    for (let i = 0; i < this.entries.length; i += 1) {
      const candidate = this.entries[i]!;
      const sim = jaccardSimilarity(qGrams, candidate.trigrams);
      if (sim > bestSim) {
        bestSim = sim;
        bestEntry = candidate;
      }
    }

    const latencyMs = Math.max(0, performance.now() - t0);
    this.totalLookupMs += latencyMs;
    if (bestEntry && bestSim >= minSimilarity) {
      this.hits += 1;
      return Object.freeze({ hit: true, similarity: bestSim, latencyMs, entry: bestEntry });
    }

    this.misses += 1;
    return Object.freeze({ hit: false, similarity: bestSim, latencyMs });
  }

  public getStats(): {
    readonly size: number;
    readonly hits: number;
    readonly misses: number;
    readonly hitRate: number;
    readonly avgLookupMs: number;
  } {
    const total = this.hits + this.misses;
    return Object.freeze({
      size: this.entries.length,
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? this.hits / total : 0,
      avgLookupMs: total > 0 ? this.totalLookupMs / total : 0,
    });
  }

  public exportTable(): readonly CrystallizedEntry<T>[] {
    return Object.freeze([...this.entries]);
  }

  public importTable(items: readonly CrystallizedEntry<T>[]): void {
    for (const item of items) {
      this.distill(item.normalizedInput, item.solution, {
        sourceModel: item.sourceModel,
        confidence: item.confidence,
      });
    }
  }
}
