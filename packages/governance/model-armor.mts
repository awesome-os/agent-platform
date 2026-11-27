/**
 * @file packages/governance/model-armor.mts
 * @description Real-time Model Armor (prompt-injection detector, secret/PII redactor,
 * Bloom-filter OOD gate) and AgentGatewayPolicy (identity, scopes, rate-limit, token ceiling).
 *
 * Strictly `erasableSyntaxOnly: true`, zero external dependencies.
 */

// ============================================================================
// 1. Bloom-Filter Out-Of-Distribution (OOD) Gate
// ============================================================================

function fnv1a32(str: string, seed: number): number {
  let h = seed >>> 0;
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

function extractCharTrigrams(text: string): string[] {
  const norm = text.toLowerCase().replace(/\s+/g, ' ').trim();
  if (norm.length === 0) return [];
  if (norm.length <= 3) return [norm];
  const grams: string[] = [];
  for (let i = 0; i <= norm.length - 3; i += 1) {
    grams.push(norm.slice(i, i + 3));
  }
  return grams;
}

export interface BloomOodResult {
  readonly inDistribution: boolean;
  readonly hitRatio: number;
  readonly oodScore: number;
  readonly totalGrams: number;
}

/**
 * Compact multi-hash Bloom filter over character trigrams for `< 1 ms` OOD gating.
 */
export interface BloomOodGateOptions {
  readonly bitCount?: number;
  readonly numHashes?: number;
  readonly threshold?: number;
  readonly seedCorpus?: readonly string[];
}

export class BloomOodGate {
  public readonly bitCount: number;
  public readonly numHashes: number;
  public readonly threshold: number;
  private readonly bits: Uint8Array;
  private trainedCount = 0;

  constructor(opts?: BloomOodGateOptions) {
    this.bitCount = opts?.bitCount ?? 4096;
    this.numHashes = opts?.numHashes ?? 3;
    this.threshold = opts?.threshold ?? 0.35;
    this.bits = new Uint8Array(Math.ceil(this.bitCount / 8));
    if (opts?.seedCorpus) {
      for (const sample of opts.seedCorpus) {
        this.train(sample);
      }
    }
  }

  private setBit(idx: number): void {
    const byteIdx = idx >> 3;
    const bitMask = 1 << (idx & 7);
    this.bits[byteIdx] = (this.bits[byteIdx] ?? 0) | bitMask;
  }

  private getBit(idx: number): boolean {
    const byteIdx = idx >> 3;
    const bitMask = 1 << (idx & 7);
    return ((this.bits[byteIdx] ?? 0) & bitMask) !== 0;
  }

  public train(text: string): void {
    const grams = extractCharTrigrams(text);
    for (const g of grams) {
      for (let k = 0; k < this.numHashes; k += 1) {
        const h = fnv1a32(g, 0x811c9dc5 + k * 0x9e3779b9) % this.bitCount;
        this.setBit(h);
      }
    }
    if (grams.length > 0) {
      this.trainedCount += 1;
    }
  }

  public score(text: string): BloomOodResult {
    const grams = extractCharTrigrams(text);
    if (grams.length === 0 || this.trainedCount === 0) {
      return Object.freeze({
        inDistribution: this.trainedCount === 0,
        hitRatio: this.trainedCount === 0 ? 1 : 0,
        oodScore: this.trainedCount === 0 ? 0 : 1,
        totalGrams: grams.length,
      });
    }

    let hits = 0;
    for (const g of grams) {
      let present = true;
      for (let k = 0; k < this.numHashes; k += 1) {
        const h = fnv1a32(g, 0x811c9dc5 + k * 0x9e3779b9) % this.bitCount;
        if (!this.getBit(h)) {
          present = false;
          break;
        }
      }
      if (present) {
        hits += 1;
      }
    }
    const hitRatio = hits / grams.length;
    const oodScore = 1 - hitRatio;
    return Object.freeze({
      inDistribution: hitRatio >= this.threshold,
      hitRatio,
      oodScore,
      totalGrams: grams.length,
    });
  }
}

// ============================================================================
// 2. ModelArmor (Prompt-Injection Detector + Secret/PII Redactor + OOD Gate)
// ============================================================================

export type InjectionSeverity = 'high' | 'medium';

export interface InjectionMatch {
  readonly ruleId: string;
  readonly severity: InjectionSeverity;
  readonly snippet: string;
}

export interface PromptInjectionScanResult {
  readonly flagged: boolean;
  readonly matches: readonly InjectionMatch[];
}

export type RedactionKind =
  | 'GITHUB_TOKEN'
  | 'AWS_KEY'
  | 'BEARER_TOKEN'
  | 'EMAIL'
  | 'CREDIT_CARD'
  | 'SSN';

export interface RedactionResult {
  readonly redactedText: string;
  readonly redactionCount: number;
  readonly countsByKind: Readonly<Record<string, number>>;
}

export interface ArmorInspectionResult {
  readonly allowed: boolean;
  readonly blockedReason?: 'PROMPT_INJECTION' | 'OOD_REJECTED';
  readonly injection: PromptInjectionScanResult;
  readonly redaction: RedactionResult;
  readonly ood?: BloomOodResult;
}

const INJECTION_RULES: readonly { readonly id: string; readonly severity: InjectionSeverity; readonly pattern: RegExp }[] = Object.freeze([
  {
    id: 'INJ_IGNORE_PREVIOUS',
    severity: 'high',
    pattern: /\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b/i,
  },
  {
    id: 'INJ_SYSTEM_PROMPT_EXFIL',
    severity: 'high',
    pattern: /\b(?:reveal|print|dump|output|leak)\s+(?:your\s+|the\s+)?(?:system\s+prompt|developer\s+instructions?|hidden\s+rules?)\b/i,
  },
  {
    id: 'INJ_ROLE_OVERRIDE',
    severity: 'high',
    pattern: /\b(?:you\s+are\s+now\s+(?:DAN|unrestricted|root|jailbroken)|act\s+as\s+unfiltered)\b/i,
  },
  {
    id: 'INJ_DELIMITER_SMUGGLING',
    severity: 'medium',
    pattern: /<\s*\|?(?:system|im_start|endoftext)\|?\s*>|```\s*system\b/i,
  },
  {
    id: 'INJ_TOOL_EXFIL_SHELL',
    severity: 'high',
    pattern: /\b(?:exfiltrate|curl\s+[^\n]*\$\{?(?:GITHUB_TOKEN|AWS_SECRET|API_KEY)|cat\s+~\/\.ssh\/id_)/i,
  },
]);

const REDACTION_PATTERNS: readonly { readonly kind: RedactionKind; readonly pattern: RegExp }[] = Object.freeze([
  { kind: 'GITHUB_TOKEN', pattern: /\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b/g },
  { kind: 'AWS_KEY', pattern: /\b(?:AKIA|ASIA|AROA|AIDA)[0-9A-Z]{16}\b/g },
  { kind: 'BEARER_TOKEN', pattern: /\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b/gi },
  { kind: 'EMAIL', pattern: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g },
  { kind: 'SSN', pattern: /\b\d{3}-\d{2}-\d{4}\b/g },
  { kind: 'CREDIT_CARD', pattern: /\b(?:\d[ -]*?){13,19}\b/g },
]);

const DEFAULT_IN_DIST_CORPUS: readonly string[] = Object.freeze([
  'audit authentication boundaries and verify unit tests pass',
  'stamp boundary verified moe model blueprint with 64-byte alignment',
  'query atlaskv hierarchical memory cluster for tenant scope',
  'run zer0q cas lookup and replay deterministic manifest',
  'inspect hardware fit metrics memoryTotalGb cpuCores for local llm',
  'resolve error B8G016 TS1294 EADDRINUSE via lumiBERT harness',
]);

export class ModelArmor {
  public readonly oodGate: BloomOodGate;
  private readonly enforceOod: boolean;

  constructor(opts?: {
    readonly oodGate?: BloomOodGate;
    readonly enforceOod?: boolean;
  }) {
    this.oodGate =
      opts?.oodGate ??
      new BloomOodGate({
        seedCorpus: DEFAULT_IN_DIST_CORPUS,
        threshold: 0.25,
      });
    this.enforceOod = opts?.enforceOod ?? false;
  }

  public detectPromptInjection(input: string): PromptInjectionScanResult {
    const matches: InjectionMatch[] = [];
    for (const rule of INJECTION_RULES) {
      const m = rule.pattern.exec(input);
      if (m) {
        matches.push({
          ruleId: rule.id,
          severity: rule.severity,
          snippet: m[0].slice(0, 80),
        });
      }
    }
    return Object.freeze({
      flagged: matches.length > 0,
      matches: Object.freeze(matches),
    });
  }

  public redactSecrets(text: string): RedactionResult {
    let redactedText = text;
    let redactionCount = 0;
    const countsByKind: Record<string, number> = {};

    for (const { kind, pattern } of REDACTION_PATTERNS) {
      const re = new RegExp(pattern.source, pattern.flags);
      redactedText = redactedText.replace(re, (match) => {
        if (kind === 'CREDIT_CARD') {
          const digits = match.replace(/\D/g, '');
          if (digits.length < 13 || digits.length > 19) {
            return match;
          }
        }
        redactionCount += 1;
        countsByKind[kind] = (countsByKind[kind] ?? 0) + 1;
        return `[REDACTED:${kind}]`;
      });
    }

    return Object.freeze({
      redactedText,
      redactionCount,
      countsByKind: Object.freeze(countsByKind),
    });
  }

  public checkOod(input: string): BloomOodResult {
    return this.oodGate.score(input);
  }

  public inspectInput(input: string, opts?: { readonly checkOod?: boolean }): ArmorInspectionResult {
    const redaction = this.redactSecrets(input);
    const injection = this.detectPromptInjection(redaction.redactedText);
    const shouldCheckOod = opts?.checkOod ?? this.enforceOod;
    const ood = shouldCheckOod ? this.checkOod(redaction.redactedText) : undefined;

    if (injection.flagged) {
      return Object.freeze({
        allowed: false,
        blockedReason: 'PROMPT_INJECTION',
        injection,
        redaction,
        ...(ood ? { ood } : {}),
      });
    }
    if (ood && !ood.inDistribution) {
      return Object.freeze({
        allowed: false,
        blockedReason: 'OOD_REJECTED',
        injection,
        redaction,
        ood,
      });
    }
    return Object.freeze({
      allowed: true,
      injection,
      redaction,
      ...(ood ? { ood } : {}),
    });
  }
}

// ============================================================================
// 3. AgentGatewayPolicy (Identity, Scopes, Rate-Limit, Token Ceiling)
// ============================================================================

export interface AgentGatewayConfig {
  readonly maxRequestsPerWindow: number;
  readonly windowMs: number;
  readonly maxTokensPerSession: number;
  readonly allowedAgentIds?: readonly string[];
}

export interface GatewayRequestContext {
  readonly agentId: string;
  readonly tenantId: string;
  readonly grantedScopes: readonly string[];
  readonly requiredScopes: readonly string[];
  readonly requestedTokens: number;
  readonly nowMs?: number;
}

export interface GatewayPolicyDecision {
  readonly allowed: boolean;
  readonly violations: readonly string[];
  readonly remainingTokens: number;
  readonly remainingRequests: number;
}

export class AgentGatewayPolicy {
  public readonly config: AgentGatewayConfig;
  private readonly requestTimestamps = new Map<string, number[]>();
  private readonly sessionTokens = new Map<string, number>();

  constructor(config: AgentGatewayConfig) {
    this.config = Object.freeze({ ...config });
  }

  public evaluateRequest(ctx: GatewayRequestContext): GatewayPolicyDecision {
    const now = ctx.nowMs ?? Date.now();
    const violations: string[] = [];

    // 1. Identity check
    if (!ctx.agentId.trim() || !ctx.tenantId.trim()) {
      violations.push('IDENTITY_MISSING: agentId and tenantId are required');
    } else if (
      this.config.allowedAgentIds &&
      !this.config.allowedAgentIds.includes(ctx.agentId)
    ) {
      violations.push(`IDENTITY_DENIED: agent "${ctx.agentId}" is not allowlisted`);
    }

    // 2. Scope inclusion check
    const granted = new Set(ctx.grantedScopes);
    if (!granted.has('*')) {
      for (const reqScope of ctx.requiredScopes) {
        if (!granted.has(reqScope)) {
          violations.push(`SCOPE_DENIED: missing required scope "${reqScope}"`);
        }
      }
    }

    // 3. Sliding-window rate limit
    const rateKey = `${ctx.tenantId}:${ctx.agentId}`;
    const windowStart = now - this.config.windowMs;
    const prevTimes = (this.requestTimestamps.get(rateKey) ?? []).filter((ts) => ts > windowStart);
    if (prevTimes.length >= this.config.maxRequestsPerWindow) {
      violations.push(
        `RATE_LIMIT_EXCEEDED: ${prevTimes.length}/${this.config.maxRequestsPerWindow} requests in ${this.config.windowMs}ms window`,
      );
    }

    // 4. Session token ceiling
    const usedTokens = this.sessionTokens.get(rateKey) ?? 0;
    if (usedTokens + ctx.requestedTokens > this.config.maxTokensPerSession) {
      violations.push(
        `TOKEN_CEILING_EXCEEDED: ${usedTokens + ctx.requestedTokens} > ${this.config.maxTokensPerSession}`,
      );
    }

    const allowed = violations.length === 0;
    if (allowed) {
      prevTimes.push(now);
      this.requestTimestamps.set(rateKey, prevTimes);
      this.sessionTokens.set(rateKey, usedTokens + Math.max(0, ctx.requestedTokens));
    }

    const curUsed = this.sessionTokens.get(rateKey) ?? 0;
    const curReqs = (this.requestTimestamps.get(rateKey) ?? []).filter((ts) => ts > windowStart).length;

    return Object.freeze({
      allowed,
      violations: Object.freeze(violations),
      remainingTokens: Math.max(0, this.config.maxTokensPerSession - curUsed),
      remainingRequests: Math.max(0, this.config.maxRequestsPerWindow - curReqs),
    });
  }
}
