/**
 * @file intersections/b8g-electron/global-ai-shim.mts
 * @description Sovereign `globalThis.ai` Local Bridge (`SovereignAiHostBridge` +
 * `createSovereignGlobalAiPreloadScript()`) replacing Chrome's closed
 * `optimization_guide` AI DLLs with zero external dependencies.
 *
 * - Strictly `erasableSyntaxOnly: true` and ZERO external npm or `node:*` dependencies.
 * - Routes error-like prompts through `DevOpsErrorHarness` (`lumiBERT` 1:1 auto-repair).
 * - Routes general prompts through a deterministic sovereign local MoE router envelope.
 * - Streams tokenized responses via pure Web `ReadableStream<string>` with backpressure.
 */

import {
  DevOpsErrorHarness,
  type ResolveErrorOptions,
  type ResolveErrorResult,
} from '../devops-harness/index.mts';

// ============================================================================
// 1. Public Contracts & Types (erasableSyntaxOnly: true)
// ============================================================================

export interface SovereignAiCapabilities {
  readonly available: 'readily';
  readonly provider: 'b8g-sovereign-local';
  readonly models: readonly ['lumiBERT-local', 'b8g-moe-router'];
  readonly closedGoogleDlls: 0;
}

export interface SovereignAiPromptOptions {
  readonly systemPrompt?: string;
  readonly threshold?: number;
}

export type SovereignAiModelId = 'lumiBERT-local' | 'b8g-moe-router';

export type SovereignAiResponseSource =
  | 'devops-error-harness'
  | 'b8g-moe-router-escalation'
  | 'sovereign-local';

export interface SovereignAiPromptResponse {
  readonly text: string;
  readonly model: SovereignAiModelId;
  readonly latencyMs: number;
  readonly source: SovereignAiResponseSource;
  readonly resolution?: ResolveErrorResult;
}

export interface SovereignAiHostBridgeOptions {
  readonly harness?: DevOpsErrorHarness;
  readonly threshold?: number;
}

export type SovereignAiIpcRequest =
  | { readonly method: 'capabilities' }
  | {
      readonly method: 'prompt';
      readonly input: string;
      readonly opts?: SovereignAiPromptOptions;
    }
  | {
      readonly method: 'resolveError';
      readonly rawError: string;
      readonly opts?: ResolveErrorOptions;
    };

// ============================================================================
// 2. Helpers
// ============================================================================

function nowMs(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

const ERROR_LIKE_PATTERNS: readonly RegExp[] = Object.freeze([
  /\b(?:error|err|exception|failed|failure|fatal|panic|crash|segfault)\b/i,
  /\b(?:eaddrinuse|enoent|eacces|econnrefused|epipe|etimedout)\b/i,
  /\b(?:ts\d{4}|b8g\d{3})\b/i,
  /\b(?:syntaxerror|typeerror|referenceerror|rangeerror|urierror)\b/i,
  /\b(?:depot_tools_win_toolchain|script_executable|erasablesyntaxonly|read_fonts)\b/i,
  /\b(?:stack trace|traceback|exit code|cannot find|not allowed|mismatch|lint)\b/i,
  /http\.createServer/i,
]);

export function isErrorLikePrompt(input: string): boolean {
  if (!input || input.trim().length === 0) {
    return false;
  }
  for (let i = 0; i < ERROR_LIKE_PATTERNS.length; i += 1) {
    if (ERROR_LIKE_PATTERNS[i]!.test(input)) {
      return true;
    }
  }
  return false;
}

function tokenizePreservingWhitespace(text: string): string[] {
  if (text.length === 0) {
    return [];
  }
  const matches = text.match(/\S+\s*|\s+/g);
  return matches ? matches : [text];
}

function formatAutoFixResponse(result: Extract<ResolveErrorResult, { status: 'AUTO_FIXED' }>): string {
  const { fix, cosine, latencyMs } = result;
  const lines: string[] = [
    `[AUTO_FIXED:${fix.code}] ${fix.summary}`,
    `Match: ${fix.id} (cosine=${cosine.toFixed(4)}, latency=${latencyMs.toFixed(2)}ms)`,
  ];
  if (fix.resolutionSteps.length > 0) {
    lines.push('Resolution Steps:');
    for (let i = 0; i < fix.resolutionSteps.length; i += 1) {
      lines.push(`${i + 1}. ${fix.resolutionSteps[i]!}`);
    }
  }
  if (fix.commandHint) {
    lines.push(`Command: ${fix.commandHint}`);
  }
  if (fix.verificationCheck) {
    lines.push(`Verify: ${fix.verificationCheck}`);
  }
  return lines.join('\n');
}

function formatGeneralLocalResponse(input: string, systemPrompt?: string): string {
  const normalizedInput = input.trim();
  const normalizedSystem = systemPrompt?.trim();
  if (normalizedSystem && normalizedSystem.length > 0) {
    return `[b8g-sovereign-local | b8g-moe-router] System: ${normalizedSystem} | Response: ${normalizedInput}`;
  }
  return `[b8g-sovereign-local | b8g-moe-router] Processed locally (0 closed Google DLLs): ${normalizedInput}`;
}

// ============================================================================
// 3. SovereignAiHostBridge
// ============================================================================

const CAPABILITIES: SovereignAiCapabilities = Object.freeze({
  available: 'readily',
  provider: 'b8g-sovereign-local',
  models: Object.freeze(['lumiBERT-local', 'b8g-moe-router'] as const),
  closedGoogleDlls: 0,
});

export class SovereignAiHostBridge {
  public readonly harness: DevOpsErrorHarness;

  constructor(opts?: SovereignAiHostBridgeOptions) {
    this.harness = opts?.harness ?? new DevOpsErrorHarness({ threshold: opts?.threshold });
  }

  /**
   * Advertises local sovereign AI capabilities with zero closed Google DLLs.
   */
  public capabilities(): SovereignAiCapabilities {
    return CAPABILITIES;
  }

  /**
   * Direct 1:1 error resolution via `DevOpsErrorHarness`.
   */
  public resolveError(rawError: string, opts?: ResolveErrorOptions): ResolveErrorResult {
    return this.harness.resolveError1To1(rawError, opts);
  }

  /**
   * Routes error-like prompts through `DevOpsErrorHarness` (1:1 auto-repair with `latencyMs`)
   * and general prompts through a deterministic sovereign local response envelope.
   */
  public prompt(input: string, opts?: SovereignAiPromptOptions): SovereignAiPromptResponse {
    const t0 = nowMs();

    if (isErrorLikePrompt(input)) {
      const resolution = this.harness.resolveError1To1(
        input,
        opts?.threshold !== undefined ? { threshold: opts.threshold } : undefined,
      );

      if (resolution.status === 'AUTO_FIXED') {
        return Object.freeze({
          text: formatAutoFixResponse(resolution),
          model: 'lumiBERT-local',
          latencyMs: resolution.latencyMs,
          source: 'devops-error-harness',
          resolution,
        });
      }

      const latencyMs = Math.max(resolution.latencyMs, nowMs() - t0);
      const nearestNote = resolution.bestCandidateId
        ? ` Nearest candidate: ${resolution.bestCandidateId}.`
        : '';
      const systemPrefix = opts?.systemPrompt?.trim()
        ? ` [system: ${opts.systemPrompt.trim()}]`
        : '';

      return Object.freeze({
        text: `[ESCALATE_TO_LLM]${systemPrefix} ${resolution.reason}${nearestNote}`,
        model: 'b8g-moe-router',
        latencyMs,
        source: 'b8g-moe-router-escalation',
        resolution,
      });
    }

    const text = formatGeneralLocalResponse(input, opts?.systemPrompt);
    const latencyMs = Math.max(0, nowMs() - t0);

    return Object.freeze({
      text,
      model: 'b8g-moe-router',
      latencyMs,
      source: 'sovereign-local',
    });
  }

  /**
   * Returns a pure Web `ReadableStream<string>` yielding tokenized chunks on demand
   * (`pull`-driven backpressure).
   */
  public promptStreaming(input: string, opts?: SovereignAiPromptOptions): ReadableStream<string> {
    let tokens: string[] | null = null;
    let cursor = 0;

    return new ReadableStream<string>({
      pull: (controller) => {
        if (tokens === null) {
          const response = this.prompt(input, opts);
          tokens = tokenizePreservingWhitespace(response.text);
        }
        if (cursor >= tokens.length) {
          controller.close();
          return;
        }
        controller.enqueue(tokens[cursor]!);
        cursor += 1;
      },
      cancel: () => {
        if (tokens !== null) {
          cursor = tokens.length;
        }
      },
    });
  }

  /**
   * Handles structured IPC invoke payloads from the renderer preload script.
   */
  public handleIpcRequest(
    request: SovereignAiIpcRequest,
  ): SovereignAiCapabilities | SovereignAiPromptResponse | ResolveErrorResult {
    switch (request.method) {
      case 'capabilities':
        return this.capabilities();
      case 'prompt':
        return this.prompt(request.input, request.opts);
      case 'resolveError':
        return this.resolveError(request.rawError, request.opts);
      default: {
        const neverReq: never = request;
        throw new Error(`[b8g:ai] Unsupported IPC method: ${JSON.stringify(neverReq)}`);
      }
    }
  }
}

// ============================================================================
// 4. Renderer Preload Script Generator
// ============================================================================

/**
 * Generates a self-contained JS preload snippet that attaches
 * `globalThis.ai = { capabilities, createTextSession, resolveError }`
 * inside any `Sectium` / `b8g-electron` renderer window, routing calls over
 * `window.__b8gElectron.ipcRenderer.invoke(ipcChannel, ...)` with a graceful
 * local fallback when running standalone.
 */
export function createSovereignGlobalAiPreloadScript(ipcChannel: string = 'b8g:ai'): string {
  const channelLiteral = JSON.stringify(ipcChannel);

  return `(() => {
  const IPC_CHANNEL = ${channelLiteral};

  const LOCAL_CAPABILITIES = Object.freeze({
    available: 'readily',
    provider: 'b8g-sovereign-local',
    models: Object.freeze(['lumiBERT-local', 'b8g-moe-router']),
    closedGoogleDlls: 0,
  });

  function getIpcRenderer() {
    if (
      typeof window !== 'undefined' &&
      window &&
      window.__b8gElectron &&
      window.__b8gElectron.ipcRenderer &&
      typeof window.__b8gElectron.ipcRenderer.invoke === 'function'
    ) {
      return window.__b8gElectron.ipcRenderer;
    }
    if (
      typeof globalThis !== 'undefined' &&
      globalThis.__b8gElectron &&
      globalThis.__b8gElectron.ipcRenderer &&
      typeof globalThis.__b8gElectron.ipcRenderer.invoke === 'function'
    ) {
      return globalThis.__b8gElectron.ipcRenderer;
    }
    return null;
  }

  function tokenize(text) {
    if (!text) return [];
    const parts = String(text).match(/\\S+\\s*|\\s+/g);
    return parts ? parts : [String(text)];
  }

  async function capabilities() {
    const ipc = getIpcRenderer();
    if (ipc) {
      return await ipc.invoke(IPC_CHANNEL, { method: 'capabilities' });
    }
    return LOCAL_CAPABILITIES;
  }

  async function resolveError(rawError, opts) {
    const ipc = getIpcRenderer();
    if (ipc) {
      return await ipc.invoke(IPC_CHANNEL, {
        method: 'resolveError',
        rawError: String(rawError),
        opts: opts || undefined,
      });
    }
    return Object.freeze({
      status: 'ESCALATE_TO_LLM',
      cosine: 0,
      reason: 'Standalone renderer fallback (no host IPC bridge attached).',
      latencyMs: 0,
    });
  }

  async function createTextSession(sessionOpts) {
    const baseOpts = sessionOpts && typeof sessionOpts === 'object' ? { ...sessionOpts } : {};
    let destroyed = false;

    function assertAlive() {
      if (destroyed) {
        throw new Error('[b8g:ai] Text session has been destroyed.');
      }
    }

    async function promptEnvelope(input, callOpts) {
      assertAlive();
      const mergedOpts = {
        ...baseOpts,
        ...(callOpts && typeof callOpts === 'object' ? callOpts : {}),
      };
      const ipc = getIpcRenderer();
      if (ipc) {
        return await ipc.invoke(IPC_CHANNEL, {
          method: 'prompt',
          input: String(input),
          opts: mergedOpts,
        });
      }
      const sys = mergedOpts.systemPrompt ? 'System: ' + String(mergedOpts.systemPrompt) + ' | ' : '';
      return Object.freeze({
        text: '[b8g-sovereign-local | standalone] ' + sys + String(input),
        model: 'b8g-moe-router',
        latencyMs: 0,
        source: 'sovereign-local',
      });
    }

    async function prompt(input, callOpts) {
      const res = await promptEnvelope(input, callOpts);
      return typeof res === 'string' ? res : res.text;
    }

    function promptStreaming(input, callOpts) {
      assertAlive();
      let tokens = null;
      let cursor = 0;

      return new ReadableStream({
        async pull(controller) {
          if (tokens === null) {
            const text = await prompt(input, callOpts);
            tokens = tokenize(text);
          }
          if (cursor >= tokens.length) {
            controller.close();
            return;
          }
          controller.enqueue(tokens[cursor]);
          cursor += 1;
        },
        cancel() {
          if (tokens !== null) {
            cursor = tokens.length;
          }
        },
      });
    }

    return Object.freeze({
      prompt,
      promptEnvelope,
      promptStreaming,
      clone() {
        assertAlive();
        return createTextSession(baseOpts);
      },
      destroy() {
        destroyed = true;
      },
    });
  }

  const sovereignAi = Object.freeze({
    capabilities,
    createTextSession,
    resolveError,
  });

  Object.defineProperty(globalThis, 'ai', {
    value: sovereignAi,
    writable: false,
    configurable: true,
    enumerable: true,
  });
})();`;
}
