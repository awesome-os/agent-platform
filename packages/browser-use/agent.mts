/**
 * @file packages/browser-use/agent.mts
 * @description BrowserUseAgent autonomous loop with bounded step history,
 * Zero-Q (`SKIMMER`) input-digest caching hook, step-cap & token/wall-time
 * budget guards, secret redaction, and structured NDJSON action receipts.
 *
 * Strictly `erasableSyntaxOnly: true`, Node 22+ built-ins only.
 */

import { performance } from 'node:perf_hooks';
import {
  type Zer0qJobManifest,
  type Zer0qRunner,
  computeInputDigest,
} from '../zer0q-substrate/index.mts';
import { redactString, redactValue } from '../zer0q-substrate/secret-redactor.mts';
import {
  type BrowserActionName,
  type BrowserActionResult,
  type BrowserObservation,
  BrowserUseController,
} from './controller.mts';
import { type DomSerializerOptions } from './dom-serializer.mts';

export type BrowserAgentDecision =
  | { readonly action: 'navigate'; readonly url: string; readonly tokensUsed?: number }
  | { readonly action: 'click'; readonly target: number | string; readonly tokensUsed?: number }
  | {
      readonly action: 'type';
      readonly target: number | string;
      readonly text: string;
      readonly clear?: boolean;
      readonly submit?: boolean;
      readonly tokensUsed?: number;
    }
  | {
      readonly action: 'scroll';
      readonly direction: 'up' | 'down' | 'top' | 'bottom' | { readonly deltaX?: number; readonly deltaY: number };
      readonly amountPx?: number;
      readonly tokensUsed?: number;
    }
  | {
      readonly action: 'extract';
      readonly selector?: string;
      readonly maxChars?: number;
      readonly format?: 'text' | 'dom-map';
      readonly tokensUsed?: number;
    }
  | {
      readonly action: 'screenshot';
      readonly fullPage?: boolean;
      readonly format?: 'png' | 'jpeg';
      readonly tokensUsed?: number;
    }
  | { readonly action: 'done'; readonly result: unknown; readonly tokensUsed?: number };

export interface BrowserHistoryStep {
  readonly step: number;
  readonly url: string;
  readonly domHash: string;
  readonly action: BrowserActionName | 'done';
  readonly target?: string;
  readonly ok: boolean;
  readonly summary: string;
  readonly tokensUsed: number;
  readonly latencyMs: number;
}

export interface BrowserPlannerContext {
  readonly task: string;
  readonly step: number;
  readonly maxSteps: number;
  readonly observation: BrowserObservation;
  readonly history: readonly BrowserHistoryStep[];
}

export type BrowserUsePlanner = (
  ctx: BrowserPlannerContext,
) => Promise<BrowserAgentDecision> | BrowserAgentDecision;

export interface BrowserUseReceipt {
  readonly seq: number;
  readonly ts: number;
  readonly runId: string;
  readonly step: number;
  readonly action: BrowserActionName | 'done' | 'cas_hit' | 'guard_stop';
  readonly target?: string;
  readonly status: 'OK' | 'CACHE_HIT' | 'ERROR' | 'BUDGET_STOP';
  readonly cached: boolean;
  readonly tokensUsed: number;
  readonly latencyMs: number;
  readonly inputDigest: string;
  readonly url: string;
  readonly domHash: string;
  readonly detail?: string;
}

export interface BrowserUseAgentOptions {
  readonly controller: BrowserUseController;
  readonly zer0qRunner?: Zer0qRunner;
  readonly model?: string;
  readonly maxSteps?: number;
  readonly maxHistorySteps?: number;
  readonly maxTokens?: number;
  readonly maxWallMs?: number;
  readonly serializerOptions?: DomSerializerOptions;
  readonly onReceipt?: (receipt: BrowserUseReceipt, ndjsonLine: string) => void;
}

export interface BrowserUseRunResult {
  readonly status: 'COMPLETED' | 'CACHE_HIT' | 'MAX_STEPS_EXCEEDED' | 'BUDGET_EXCEEDED' | 'FAILED';
  readonly cached: boolean;
  readonly runId: string;
  readonly inputDigest: string;
  readonly stepsExecuted: number;
  readonly tokensUsed: number;
  readonly latencyMs: number;
  readonly finalUrl: string;
  readonly result: unknown;
  readonly history: readonly BrowserHistoryStep[];
  readonly receipts: readonly BrowserUseReceipt[];
  readonly ndjson: string;
}

function summarizeActionResult(res: BrowserActionResult): string {
  if (!res.ok) {
    return res.error ?? 'error';
  }
  if (res.action === 'extract') {
    const data = res.data as { readonly content?: string } | undefined;
    return `extracted ${data?.content?.length ?? 0} chars`;
  }
  if (res.action === 'screenshot') {
    const data = res.data as { readonly bytes?: number } | undefined;
    return `screenshot ${data?.bytes ?? 0} bytes`;
  }
  return `${res.action}${res.target ? `(${res.target})` : ''} ok`;
}

export class BrowserUseAgent {
  private readonly controller: BrowserUseController;
  private readonly zer0qRunner?: Zer0qRunner;
  private readonly model: string;
  private readonly defaultMaxSteps: number;
  private readonly maxHistorySteps: number;
  private readonly defaultMaxTokens: number;
  private readonly defaultMaxWallMs: number;
  private readonly serializerOptions?: DomSerializerOptions;
  private readonly onReceipt?: (receipt: BrowserUseReceipt, ndjsonLine: string) => void;

  constructor(options: BrowserUseAgentOptions) {
    this.controller = options.controller;
    this.zer0qRunner = options.zer0qRunner;
    this.model = options.model ?? 'b8g-xenon-1b';
    this.defaultMaxSteps = Math.max(1, options.maxSteps ?? 12);
    this.maxHistorySteps = Math.max(1, options.maxHistorySteps ?? 8);
    this.defaultMaxTokens = Math.max(1, options.maxTokens ?? 50_000);
    this.defaultMaxWallMs = Math.max(1, options.maxWallMs ?? 30_000);
    this.serializerOptions = options.serializerOptions;
    this.onReceipt = options.onReceipt;
  }

  public async run(opts: {
    readonly task: string;
    readonly startUrl?: string;
    readonly planner: BrowserUsePlanner;
    readonly maxSteps?: number;
    readonly maxTokens?: number;
    readonly maxWallMs?: number;
    readonly cachePolicy?: 'zero-q-lookup-or-run' | 'bypass';
  }): Promise<BrowserUseRunResult> {
    const t0 = performance.now();
    const maxSteps = Math.max(1, opts.maxSteps ?? this.defaultMaxSteps);
    const maxTokens = Math.max(1, opts.maxTokens ?? this.defaultMaxTokens);
    const maxWallMs = Math.max(1, opts.maxWallMs ?? this.defaultMaxWallMs);
    const cachePolicy = opts.cachePolicy ?? 'zero-q-lookup-or-run';

    const startUrl = opts.startUrl ?? this.controller.getCurrentUrl();
    const manifest: Zer0qJobManifest = {
      apiVersion: 'zer0q.awesome-os/v1',
      name: 'browser-use-task',
      inputs: {
        agentsMd: 'pkg:browser-use/v1',
        skills: ['browser-use', 'sectium-device'],
        workspace: startUrl,
        prompt: `${opts.task} [maxSteps=${maxSteps}]`,
      },
      execution: {
        model: this.model,
        cachePolicy,
        maxTokens,
        maxWallMs,
        maxToolCalls: maxSteps,
      },
    };

    const inputDigest = this.zer0qRunner
      ? this.zer0qRunner.computeManifestDigest(manifest)
      : computeInputDigest(manifest);

    const receipts: BrowserUseReceipt[] = [];
    const ndjsonLines: string[] = [];

    const emitReceipt = (partial: Omit<BrowserUseReceipt, 'seq' | 'ts'>): void => {
      const sanitizedDetail =
        partial.detail !== undefined ? redactString(partial.detail).redacted : undefined;
      const receipt: BrowserUseReceipt = Object.freeze({
        ...partial,
        ...(sanitizedDetail !== undefined ? { detail: sanitizedDetail } : {}),
        seq: receipts.length + 1,
        ts: Date.now(),
      });
      receipts.push(receipt);
      const line = JSON.stringify(receipt);
      ndjsonLines.push(line);
      if (this.onReceipt) {
        this.onReceipt(receipt, line);
      }
    };

    interface CachedAgentPayload {
      readonly result: unknown;
      readonly finalUrl: string;
      readonly stepsExecuted: number;
      readonly history: readonly BrowserHistoryStep[];
    }

    if (this.zer0qRunner && cachePolicy === 'zero-q-lookup-or-run') {
      const hit = this.zer0qRunner.lookup<CachedAgentPayload>(manifest);
      if (hit.hit && hit.output && hit.runId) {
        emitReceipt({
          runId: hit.runId,
          step: 0,
          action: 'cas_hit',
          status: 'CACHE_HIT',
          cached: true,
          tokensUsed: 0,
          latencyMs: hit.latencyMs,
          inputDigest,
          url: hit.output.finalUrl,
          domHash: hit.output.history[hit.output.history.length - 1]?.domHash ?? '',
          detail: 'ZER0Q SKIMMER short-circuit cache hit',
        });
        return Object.freeze({
          status: 'CACHE_HIT',
          cached: true,
          runId: hit.runId,
          inputDigest,
          stepsExecuted: hit.output.stepsExecuted,
          tokensUsed: 0,
          latencyMs: Math.max(0, performance.now() - t0),
          finalUrl: hit.output.finalUrl,
          result: hit.output.result,
          history: hit.output.history,
          receipts: Object.freeze(receipts),
          ndjson: ndjsonLines.join('\n'),
        });
      }
    }

    const executeLoop = async (runId: string): Promise<{
      readonly status: BrowserUseRunResult['status'];
      readonly result: unknown;
      readonly finalUrl: string;
      readonly stepsExecuted: number;
      readonly tokensUsed: number;
      readonly history: readonly BrowserHistoryStep[];
    }> => {
      if (opts.startUrl) {
        const navRes = await this.controller.navigate(opts.startUrl);
        emitReceipt({
          runId,
          step: 0,
          action: 'navigate',
          target: opts.startUrl,
          status: 'OK',
          cached: false,
          tokensUsed: 0,
          latencyMs: navRes.latencyMs,
          inputDigest,
          url: navRes.url,
          domHash: '',
          detail: 'initial navigation',
        });
      }

      const history: BrowserHistoryStep[] = [];
      let totalTokens = 0;
      let stepsExecuted = 0;
      let finalResult: unknown = null;
      let status: BrowserUseRunResult['status'] = 'MAX_STEPS_EXCEEDED';

      const pushHistory = (entry: BrowserHistoryStep): void => {
        history.push(entry);
        if (history.length > this.maxHistorySteps) {
          history.splice(0, history.length - this.maxHistorySteps);
        }
      };

      for (let step = 1; step <= maxSteps; step += 1) {
        const elapsed = performance.now() - t0;
        if (elapsed > maxWallMs) {
          status = 'BUDGET_EXCEEDED';
          emitReceipt({
            runId,
            step,
            action: 'guard_stop',
            status: 'BUDGET_STOP',
            cached: false,
            tokensUsed: 0,
            latencyMs: elapsed,
            inputDigest,
            url: this.controller.getCurrentUrl(),
            domHash: '',
            detail: `WALL_TIME_EXCEEDED (${elapsed.toFixed(1)} ms > ${maxWallMs} ms)`,
          });
          break;
        }

        const observation = await this.controller.observe(this.serializerOptions);
        const stepStart = performance.now();
        const decision = await opts.planner({
          task: opts.task,
          step,
          maxSteps,
          observation,
          history: Object.freeze([...history]),
        });

        const stepTokens = Math.max(0, decision.tokensUsed ?? 0);
        if (totalTokens + stepTokens > maxTokens) {
          status = 'BUDGET_EXCEEDED';
          emitReceipt({
            runId,
            step,
            action: 'guard_stop',
            status: 'BUDGET_STOP',
            cached: false,
            tokensUsed: stepTokens,
            latencyMs: Math.max(0, performance.now() - stepStart),
            inputDigest,
            url: observation.url,
            domHash: observation.domHash,
            detail: `TOKEN_CEILING_EXCEEDED (${totalTokens + stepTokens} > ${maxTokens})`,
          });
          break;
        }

        totalTokens += stepTokens;
        stepsExecuted = step;

        if (decision.action === 'done') {
          finalResult = decision.result;
          status = 'COMPLETED';
          const latencyMs = Math.max(0, performance.now() - stepStart);
          pushHistory({
            step,
            url: observation.url,
            domHash: observation.domHash,
            action: 'done',
            ok: true,
            summary: 'completed',
            tokensUsed: stepTokens,
            latencyMs,
          });
          emitReceipt({
            runId,
            step,
            action: 'done',
            status: 'OK',
            cached: false,
            tokensUsed: stepTokens,
            latencyMs,
            inputDigest,
            url: observation.url,
            domHash: observation.domHash,
            detail: 'task completed',
          });
          break;
        }

        let actionResult: BrowserActionResult;
        if (decision.action === 'navigate') {
          actionResult = await this.controller.navigate(decision.url);
        } else if (decision.action === 'click') {
          actionResult = await this.controller.click(decision.target);
        } else if (decision.action === 'type') {
          actionResult = await this.controller.type(decision.target, decision.text, {
            clear: decision.clear,
            submit: decision.submit,
          });
        } else if (decision.action === 'scroll') {
          actionResult = await this.controller.scroll(decision.direction, decision.amountPx);
        } else if (decision.action === 'extract') {
          actionResult = await this.controller.extract({
            selector: decision.selector,
            maxChars: decision.maxChars,
            format: decision.format,
          });
          finalResult = actionResult.data;
        } else {
          actionResult = await this.controller.screenshot({
            fullPage: decision.fullPage,
            format: decision.format,
          });
          finalResult = actionResult.data;
        }

        const stepLatency = Math.max(0, performance.now() - stepStart);
        const summary = summarizeActionResult(actionResult);
        pushHistory({
          step,
          url: actionResult.url,
          domHash: observation.domHash,
          action: decision.action,
          ...(actionResult.target !== undefined ? { target: actionResult.target } : {}),
          ok: actionResult.ok,
          summary,
          tokensUsed: stepTokens,
          latencyMs: stepLatency,
        });
        emitReceipt({
          runId,
          step,
          action: decision.action,
          ...(actionResult.target !== undefined ? { target: actionResult.target } : {}),
          status: actionResult.ok ? 'OK' : 'ERROR',
          cached: false,
          tokensUsed: stepTokens,
          latencyMs: stepLatency,
          inputDigest,
          url: actionResult.url,
          domHash: observation.domHash,
          detail: summary,
        });
      }

      return {
        status,
        result: redactValue(finalResult).redacted,
        finalUrl: this.controller.getCurrentUrl(),
        stepsExecuted,
        tokensUsed: totalTokens,
        history: Object.freeze([...history]),
      };
    };

    if (this.zer0qRunner) {
      let loopOutcome: Awaited<ReturnType<typeof executeLoop>> | undefined;
      const runRes = await this.zer0qRunner.run<CachedAgentPayload>(manifest, async (ctx) => {
        loopOutcome = await executeLoop(ctx.runId);
        return {
          output: {
            result: loopOutcome.result,
            finalUrl: loopOutcome.finalUrl,
            stepsExecuted: loopOutcome.stepsExecuted,
            history: loopOutcome.history,
          },
          tokensUsed: loopOutcome.tokensUsed,
        };
      });

      const outcome = loopOutcome ?? {
        status: 'CACHE_HIT' as const,
        result: runRes.output.result,
        finalUrl: runRes.output.finalUrl,
        stepsExecuted: runRes.output.stepsExecuted,
        tokensUsed: runRes.tokensUsed,
        history: runRes.output.history,
      };

      return Object.freeze({
        status: outcome.status,
        cached: runRes.cached,
        runId: runRes.runId,
        inputDigest: runRes.inputDigest,
        stepsExecuted: outcome.stepsExecuted,
        tokensUsed: outcome.tokensUsed,
        latencyMs: Math.max(0, performance.now() - t0),
        finalUrl: outcome.finalUrl,
        result: outcome.result,
        history: outcome.history,
        receipts: Object.freeze(receipts),
        ndjson: ndjsonLines.join('\n'),
      });
    }

    const localRunId = `browser-${inputDigest.slice(0, 12)}`;
    const outcome = await executeLoop(localRunId);
    return Object.freeze({
      status: outcome.status,
      cached: false,
      runId: localRunId,
      inputDigest,
      stepsExecuted: outcome.stepsExecuted,
      tokensUsed: outcome.tokensUsed,
      latencyMs: Math.max(0, performance.now() - t0),
      finalUrl: outcome.finalUrl,
      result: outcome.result,
      history: outcome.history,
      receipts: Object.freeze(receipts),
      ndjson: ndjsonLines.join('\n'),
    });
  }
}
