/**
 * @file intersections/b8g-devops-harness/index.mts
 * @description 1:1 Error-to-Solution `devops-harness-architecture` powered by
 * `sections/lumi-bert` + deterministic verifier gate + `crystallizeFix()` learning loop
 * + persona-gated `intersections/b8g-devops` escalation.
 *
 * - Strictly `erasableSyntaxOnly: true` and ZERO external npm or `node:*` dependencies.
 */

import {
  defineSection,
  f,
  intersect,
  s,
  type ExecutionContext,
} from '../schema/index.mts';
import {
  LumiBertEncoder,
  LumiBertSection,
  type LumiBertConfig,
} from '../lumi-bert/index.mts';
import {
  B8G_DEVOPS_CONTEXT,
  B8gDevOpsIntersection,
  emitDevOpsEvent,
} from './devops.mts';

// ============================================================================
// 1. Declarative StealifyLang Section & Intersection Contracts
// ============================================================================

export const ERROR_FIX_CAS_CONTEXT: ExecutionContext = Object.freeze({
  boundary: 'in-process',
  display: 'headless',
  lifetime: 'singleton-service',
  transport: 'arena-sab',
  capability: Object.freeze({
    tenantId: 'local-user',
    scopes: Object.freeze(['devops.harness', 'lumibert.embed']),
    epoch: 1,
    privateData: true,
  }),
  locality: 'local-arena',
  persona: 'devops',
});

export const ErrorFixCasSection = defineSection({
  id: 'b8g/error-fix-cas',
  version: '1.0.0',
  substrate: s.sabRing({ ringBytes: 65536 }),
  context: ERROR_FIX_CAS_CONTEXT,
  fields: {
    harnessTotalQueries: f.int32().default(0),
    harnessAutoFixedCount: f.int32().default(0),
    harnessEscalatedCount: f.int32().default(0),
    harnessCrystallizedCount: f.int32().default(0),
    harnessLastFixId: f.string({ maxBytes: 64 }).default(''),
    harnessLastCosine: f.float64().default(0),
  },
});

export const B8gDevOpsHarnessIntersection = intersect('b8g/devops-harness', [
  LumiBertSection,
  ErrorFixCasSection,
  B8gDevOpsIntersection,
] as const);

// ============================================================================
// 2. 1:1 Error-to-Solution Types & Pre-Seeded Awesome-OS Knowledge Base
// ============================================================================

export interface ErrorFixRecord {
  readonly id: string;
  readonly code: string;
  readonly category: string;
  readonly canonicalError: string;
  readonly summary: string;
  readonly resolutionSteps: readonly string[];
  readonly commandHint?: string;
  readonly verificationCheck?: string;
  readonly crystallizedAt?: number;
}

export interface ResolveErrorOptions {
  readonly threshold?: number;
  readonly verifyFix?: (fix: ErrorFixRecord) => boolean;
  readonly callerCtx?: ExecutionContext;
}

export type ResolveErrorResult =
  | {
      readonly status: 'AUTO_FIXED';
      readonly cosine: number;
      readonly fix: ErrorFixRecord;
      readonly latencyMs: number;
    }
  | {
      readonly status: 'ESCALATE_TO_LLM';
      readonly cosine: number;
      readonly reason: string;
      readonly bestCandidateId?: string;
      readonly latencyMs: number;
    };

export interface HarnessBenchmarkReport {
  readonly iterations: number;
  readonly avgLatencyMs: number;
  readonly p95LatencyMs: number;
  readonly maxLatencyMs: number;
  readonly autoFixedRate: number;
  readonly isPretrainedWeightsLoaded: boolean;
}

export const DEFAULT_MATCH_THRESHOLD = 0.45;

export const SEEDED_ERROR_FIXES: readonly ErrorFixRecord[] = Object.freeze([
  Object.freeze({
    id: 'fix-eaddrinuse',
    code: 'EADDRINUSE',
    category: 'runtime-ipc',
    canonicalError:
      'Error: listen EADDRINUSE: address already in use 127.0.0.1:8080 port already bound',
    summary:
      'Port is already occupied by a stale listener; reclaim the port or migrate local IPC to B8G SAB/OS-pipe.',
    resolutionSteps: Object.freeze([
      'Inspect owning PID on Windows: Get-NetTCPConnection -LocalPort <port> | Select-Object OwningProcess',
      'Terminate stale listener or bind ephemeral port 0 in tests.',
      'For internal local IPC, prefer @awesome-os/stealify-schema SAB ring or os-pipe instead of localhost TCP.',
    ]),
    commandHint: 'Get-NetTCPConnection -LocalPort <port> | Select-Object OwningProcess',
    verificationCheck: 'Listener binds cleanly with zero EADDRINUSE errors.',
  }),
  Object.freeze({
    id: 'fix-depot-tools-win-toolchain',
    code: 'DEPOT_TOOLS_WIN_TOOLCHAIN',
    category: 'chromium-build',
    canonicalError:
      'DEPOT_TOOLS_WIN_TOOLCHAIN=0 vs_toolchain.py failed to find Visual Studio toolchain on Windows',
    summary:
      'Chromium depot_tools requires DEPOT_TOOLS_WIN_TOOLCHAIN=0 when building with a local Windows SDK / MSVC installation.',
    resolutionSteps: Object.freeze([
      'Set $env:DEPOT_TOOLS_WIN_TOOLCHAIN="0" in PowerShell before invoking gn or gclient.',
      'Ensure Visual Studio C++ Desktop workload and Windows 11 SDK are installed and GYP_MSVS_OVERRIDE_PATH is valid.',
    ]),
    commandHint: '$env:DEPOT_TOOLS_WIN_TOOLCHAIN="0"',
    verificationCheck: 'gn gen succeeds without vs_toolchain.py credential prompt.',
  }),
  Object.freeze({
    id: 'fix-gn-script-executable',
    code: 'GN_SCRIPT_EXECUTABLE',
    category: 'chromium-build',
    canonicalError:
      'ERROR at //.gn:18:21: This is not a string. script_executable = ["python3"] must be a string',
    summary:
      'GN `script_executable` in `.gn` must be a scalar string path, never a list.',
    resolutionSteps: Object.freeze([
      'Edit //.gn and set `script_executable = "python3.exe"` (or absolute path string), removing array brackets `[...]`.',
      'Re-run `gn gen` to verify root configuration parses cleanly.',
    ]),
    commandHint: 'script_executable = "python3.exe"',
    verificationCheck: 'gn gen parses //.gn with exit code 0.',
  }),
  Object.freeze({
    id: 'fix-b8g016-http-create-server',
    code: 'B8G016',
    category: 'stealify-lint',
    canonicalError:
      'B8G016 lint error forbidden http.createServer raw node:http localhost IPC server',
    summary:
      'B8G016 forbids raw `http.createServer` for local same-boundary IPC; use SAB rings, os-pipe, or b8g-host-bus.',
    resolutionSteps: Object.freeze([
      'Remove `http.createServer` from local IPC paths.',
      'Route same-boundary messages via `@awesome-os/stealify-schema` SAB ring (`arena-sab`) or `os-pipe`.',
      'Re-run stealify-lint to confirm 0 B8G016 violations.',
    ]),
    commandHint: '.\\bin\\stealify-lint.ps1 <file>',
    verificationCheck: 'stealify-lint reports 0 B8G016 diagnostics.',
  }),
  Object.freeze({
    id: 'fix-b8g018-html-table-columns',
    code: 'B8G018',
    category: 'web-doc-lint',
    canonicalError:
      'B8G018 HTML table column mismatch tr td th count does not match header columns',
    summary:
      'B8G018 detected an HTML table row whose cell count (including colspan) disagrees with the header column count.',
    resolutionSteps: Object.freeze([
      'Audit every <tr> in the flagged <table> so total <th>/<td> cells (accounting for colspan) match the header row.',
      'Verify with `intersections/b8g-web-doc/cli.mts <file.html>`.',
    ]),
    commandHint: '.\\bin\\node.ps1 intersections/b8g-web-doc/cli.mts <file.html>',
    verificationCheck: 'b8g-web-doc reports PASS with 0 B8G018 diagnostics.',
  }),
  Object.freeze({
    id: 'fix-b8g019-markdown-pipe',
    code: 'B8G019',
    category: 'web-doc-lint',
    canonicalError:
      'B8G019 Markdown table unescaped pipe character column count mismatch in markdown cell',
    summary:
      'B8G019 detected an unescaped `|` inside a Markdown table cell causing extra columns.',
    resolutionSteps: Object.freeze([
      'Escape literal pipe characters inside Markdown table cells as `\\|` (or wrap union types carefully).',
      'Verify with `intersections/b8g-web-doc/cli.mts <file.md>`.',
    ]),
    commandHint: '.\\bin\\node.ps1 intersections/b8g-web-doc/cli.mts <file.md>',
    verificationCheck: 'b8g-web-doc reports PASS with 0 B8G019 diagnostics.',
  }),
  Object.freeze({
    id: 'fix-erasable-syntax-only-enum',
    code: 'TS1294_ERASABLE_SYNTAX',
    category: 'typescript-syntax',
    canonicalError:
      'TS1294 erasableSyntaxOnly This syntax is not allowed when erasableSyntaxOnly is enabled enum namespace parameter property',
    summary:
      'Node 22+ native type stripping enforces `erasableSyntaxOnly: true`; TypeScript `enum`, `namespace`, and constructor parameter properties are forbidden.',
    resolutionSteps: Object.freeze([
      'Replace `enum Foo { ... }` with `export const Foo = { ... } as const; export type Foo = (typeof Foo)[keyof typeof Foo];`.',
      'Replace constructor parameter properties (`constructor(public x: number)`) with explicit class fields assigned in constructor.',
    ]),
    commandHint: '.\\bin\\node.ps1 --test <file.mts>',
    verificationCheck: 'Module loads directly in Node 22+ with zero TS1294 syntax errors.',
  }),
  Object.freeze({
    id: 'fix-chromium-read-fonts-deps',
    code: 'DEPS_READ_FONTS_DRIFT',
    category: 'chromium-rust-deps',
    canonicalError:
      'third_party/rust/read_fonts DEPS drift missing crate version mismatch Cargo.toml BUILD.gn',
    summary:
      'Chromium Rust `third_party/rust/read_fonts` crate revision drifted out of sync with DEPS / BUILD.gn.',
    resolutionSteps: Object.freeze([
      'Reconcile `third_party/rust/read_fonts` pin in `DEPS` against upstream `third_party/rust/chromium_crates_io/Cargo.lock`.',
      'Re-run `gclient sync --no-history` and regenerate GN rules via `gnrt` if needed.',
    ]),
    commandHint: 'gclient sync --no-history',
    verificationCheck: 'gn gen resolves third_party/rust/read_fonts targets cleanly.',
  }),
]);

// ============================================================================
// 3. DevOpsErrorHarness Engine
// ============================================================================

interface IndexedFixEntry {
  readonly record: ErrorFixRecord;
  readonly vector: Float32Array;
}

function nowMs(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

export interface DevOpsErrorHarnessOptions {
  readonly encoder?: LumiBertEncoder;
  readonly encoderConfig?: Partial<LumiBertConfig>;
  readonly threshold?: number;
  readonly seedDefaults?: boolean;
}

export class DevOpsErrorHarness {
  public readonly encoder: LumiBertEncoder;
  private readonly entries: IndexedFixEntry[] = [];
  private readonly defaultThreshold: number;

  constructor(opts?: DevOpsErrorHarnessOptions) {
    this.encoder = opts?.encoder ?? new LumiBertEncoder(opts?.encoderConfig);
    this.defaultThreshold = opts?.threshold ?? DEFAULT_MATCH_THRESHOLD;

    if (opts?.seedDefaults !== false) {
      for (const fix of SEEDED_ERROR_FIXES) {
        this.indexFix(fix.canonicalError, fix);
      }
    }
  }

  private indexFix(exampleText: string, fix: ErrorFixRecord): void {
    const vector = this.encoder.embed(exampleText);
    const existingIdx = this.entries.findIndex((e) => e.record.id === fix.id);
    const entry: IndexedFixEntry = { record: Object.freeze({ ...fix }), vector };
    if (existingIdx >= 0) {
      this.entries[existingIdx] = entry;
    } else {
      this.entries.push(entry);
    }
  }

  /**
   * Returns all currently indexed 1:1 error-to-solution records.
   */
  public listFixes(): readonly ErrorFixRecord[] {
    return Object.freeze(this.entries.map((e) => e.record));
  }

  /**
   * 1:1 Error-to-Solution lookup with deterministic verifier gate.
   *
   * - Embeds `rawError` via `lumiBERT`, finds top-1 cosine match.
   * - If `cosine >= threshold` AND optional `verifyFix(fix)` passes -> `AUTO_FIXED`.
   * - Otherwise emits a `P2` / `dev-error-loop` event to `b8g-devops` and returns
   *   `ESCALATE_TO_LLM`.
   */
  public resolveError1To1(
    rawError: string,
    opts?: ResolveErrorOptions,
  ): ResolveErrorResult {
    const t0 = nowMs();
    const threshold = opts?.threshold ?? this.defaultThreshold;
    const queryVec = this.encoder.embed(rawError);

    let bestCosine = -1;
    let bestEntry: IndexedFixEntry | undefined;

    for (let i = 0; i < this.entries.length; i += 1) {
      const entry = this.entries[i]!;
      const sim = this.encoder.cosineSimilarity(queryVec, entry.vector);
      if (sim > bestCosine) {
        bestCosine = sim;
        bestEntry = entry;
      }
    }

    const latencyMs = Math.max(0, nowMs() - t0);

    if (bestEntry !== undefined && bestCosine >= threshold) {
      const verified = opts?.verifyFix ? opts.verifyFix(bestEntry.record) : true;
      if (verified) {
        return Object.freeze({
          status: 'AUTO_FIXED',
          cosine: bestCosine,
          fix: bestEntry.record,
          latencyMs,
        });
      }

      const reason = `Deterministic verifier rejected candidate "${bestEntry.record.id}" (cosine=${bestCosine.toFixed(4)}).`;
      emitDevOpsEvent(
        {
          severity: 'P2',
          domain: 'dev',
          workerTarget: 'dev-error-loop',
          source: 'b8g/devops-harness',
          code: 'HARNESS_VERIFIER_REJECTED',
          message: reason,
          minPersona: 'devops',
          emitConsole: false,
          metadata: {
            candidateId: bestEntry.record.id,
            cosine: bestCosine,
            rawErrorPreview: rawError.slice(0, 160),
          },
        },
        opts?.callerCtx ?? B8G_DEVOPS_CONTEXT,
      );

      return Object.freeze({
        status: 'ESCALATE_TO_LLM',
        cosine: bestCosine,
        reason,
        bestCandidateId: bestEntry.record.id,
        latencyMs,
      });
    }

    const reason =
      bestEntry !== undefined
        ? `Top-1 cosine ${bestCosine.toFixed(4)} below threshold ${threshold.toFixed(2)} (nearest="${bestEntry.record.id}").`
        : 'No error-fix records indexed in harness CAS.';

    emitDevOpsEvent(
      {
        severity: 'P2',
        domain: 'dev',
        workerTarget: 'dev-error-loop',
        source: 'b8g/devops-harness',
        code: 'HARNESS_NOVEL_ERROR',
        message: reason,
        minPersona: 'devops',
        emitConsole: false,
        metadata: {
          cosine: bestCosine,
          threshold,
          ...(bestEntry ? { nearestCandidateId: bestEntry.record.id } : {}),
          rawErrorPreview: rawError.slice(0, 160),
        },
      },
      opts?.callerCtx ?? B8G_DEVOPS_CONTEXT,
    );

    return Object.freeze({
      status: 'ESCALATE_TO_LLM',
      cosine: bestCosine,
      reason,
      ...(bestEntry ? { bestCandidateId: bestEntry.record.id } : {}),
      latencyMs,
    });
  }

  /**
   * Crystallizes a newly solved error-to-solution pair into the harness CAS so
   * future occurrences resolve in `< 5 ms` with 0 LLM tokens.
   */
  public crystallizeFix(errorExample: string, fix: ErrorFixRecord): ErrorFixRecord {
    const stamped: ErrorFixRecord = Object.freeze({
      ...fix,
      crystallizedAt: fix.crystallizedAt ?? Date.now(),
    });
    this.indexFix(errorExample, stamped);
    return stamped;
  }

  /**
   * Benchmarks 1:1 resolution latency and hit-rate across seeded error probes.
   */
  public benchmark(iterations: number = 40): HarnessBenchmarkReport {
    const probes = [
      'Error: listen EADDRINUSE: address already in use 127.0.0.1:8890',
      'ERROR at //.gn:18:21: This is not a string. script_executable = ["python3"]',
      'B8G016 forbidden http.createServer raw localhost HTTP server',
      'B8G018 HTML table column mismatch tr td count',
      'B8G019 Markdown table unescaped pipe character',
      'TS1294 erasableSyntaxOnly enum not allowed',
      'DEPOT_TOOLS_WIN_TOOLCHAIN=0 vs_toolchain.py failed to find Visual Studio',
      'third_party/rust/read_fonts DEPS drift version mismatch',
    ];

    // Warmup JIT
    for (let w = 0; w < 4; w += 1) {
      this.resolveError1To1(probes[w % probes.length]!);
    }

    const latencies: number[] = [];
    let autoFixed = 0;
    const total = Math.max(1, iterations);

    for (let i = 0; i < total; i += 1) {
      const probe = probes[i % probes.length]!;
      const res = this.resolveError1To1(probe);
      latencies.push(res.latencyMs);
      if (res.status === 'AUTO_FIXED') {
        autoFixed += 1;
      }
    }

    latencies.sort((a, b) => a - b);
    const sum = latencies.reduce((acc, v) => acc + v, 0);
    const p95Idx = Math.min(latencies.length - 1, Math.floor(latencies.length * 0.95));

    return Object.freeze({
      iterations: total,
      avgLatencyMs: sum / total,
      p95LatencyMs: latencies[p95Idx]!,
      maxLatencyMs: latencies[latencies.length - 1]!,
      autoFixedRate: autoFixed / total,
      isPretrainedWeightsLoaded: this.encoder.isPretrainedWeightsLoaded,
    });
  }
}

// ============================================================================
// 4. Default Singleton Convenience API
// ============================================================================

export const defaultDevOpsHarness = new DevOpsErrorHarness();

export function resolveError1To1(
  rawError: string,
  opts?: ResolveErrorOptions,
): ResolveErrorResult {
  return defaultDevOpsHarness.resolveError1To1(rawError, opts);
}

export function crystallizeFix(
  errorExample: string,
  fix: ErrorFixRecord,
): ErrorFixRecord {
  return defaultDevOpsHarness.crystallizeFix(errorExample, fix);
}

export function listKnownFixes(): readonly ErrorFixRecord[] {
  return defaultDevOpsHarness.listFixes();
}
