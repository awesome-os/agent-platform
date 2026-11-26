/**
 * @file packages/zer0q-substrate/secret-redactor.mts
 * @description Mandatory pre-CAS-commit secret redaction scanner & scrubber.
 * Strips API keys, Bearer tokens, GitHub `ghp_`/`github_pat_` tokens,
 * `TWITCH_STREAM_KEY`, and PEM private key blocks before immutable CAS persistence.
 *
 * Strictly `erasableSyntaxOnly: true`, zero external runtime dependencies.
 */

export type SecretPatternKind =
  | 'PEM_PRIVATE_KEY'
  | 'GITHUB_TOKEN'
  | 'BEARER_TOKEN'
  | 'TWITCH_STREAM_KEY'
  | 'OPENAI_API_KEY'
  | 'ANTHROPIC_API_KEY'
  | 'GOOGLE_API_KEY'
  | 'AWS_ACCESS_KEY'
  | 'SLACK_TOKEN'
  | 'GENERIC_SECRET_FIELD';

export interface SecretFinding {
  readonly kind: SecretPatternKind;
  readonly path: string;
  readonly preview: string;
}

export interface RedactStringResult {
  readonly redacted: string;
  readonly findings: readonly SecretFinding[];
  readonly count: number;
}

export interface RedactValueResult<T> {
  readonly redacted: T;
  readonly findings: readonly SecretFinding[];
  readonly count: number;
}

const REDACTED_MARKER = '[REDACTED]';

interface PatternRule {
  readonly kind: SecretPatternKind;
  readonly regex: RegExp;
  readonly replace: (match: string, ...groups: string[]) => string;
}

const PATTERN_RULES: readonly PatternRule[] = [
  {
    kind: 'PEM_PRIVATE_KEY',
    regex: /-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----/g,
    replace: () => `[REDACTED:PEM_PRIVATE_KEY]`,
  },
  {
    kind: 'GITHUB_TOKEN',
    regex: /\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b/g,
    replace: () => `[REDACTED:GITHUB_TOKEN]`,
  },
  {
    kind: 'TWITCH_STREAM_KEY',
    regex: /\b(TWITCH_STREAM_KEY)\s*([:=])\s*(['"]?)([A-Za-z0-9_-]{8,})\3/gi,
    replace: (_m, key, sep, quote) => `${key}${sep}${quote}[REDACTED:TWITCH_STREAM_KEY]${quote}`,
  },
  {
    kind: 'TWITCH_STREAM_KEY',
    regex: /\blive_[0-9]{5,}_[A-Za-z0-9]{16,}\b/g,
    replace: () => `[REDACTED:TWITCH_STREAM_KEY]`,
  },
  {
    kind: 'BEARER_TOKEN',
    regex: /\b(Bearer)\s+([A-Za-z0-9\-._~+/]{12,}=*)/gi,
    replace: (_m, scheme) => `${scheme} [REDACTED:BEARER_TOKEN]`,
  },
  {
    kind: 'ANTHROPIC_API_KEY',
    regex: /\bsk-ant-[A-Za-z0-9_-]{16,}\b/g,
    replace: () => `[REDACTED:ANTHROPIC_API_KEY]`,
  },
  {
    kind: 'OPENAI_API_KEY',
    regex: /\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b/g,
    replace: () => `[REDACTED:OPENAI_API_KEY]`,
  },
  {
    kind: 'GOOGLE_API_KEY',
    regex: /\bAIza[0-9A-Za-z_-]{35}\b/g,
    replace: () => `[REDACTED:GOOGLE_API_KEY]`,
  },
  {
    kind: 'AWS_ACCESS_KEY',
    regex: /\bAKIA[0-9A-Z]{16}\b/g,
    replace: () => `[REDACTED:AWS_ACCESS_KEY]`,
  },
  {
    kind: 'SLACK_TOKEN',
    regex: /\bxox[baprs]-[0-9A-Za-z-]{10,}\b/g,
    replace: () => `[REDACTED:SLACK_TOKEN]`,
  },
  {
    kind: 'GENERIC_SECRET_FIELD',
    regex:
      /\b(api[_-]?key|secret[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|private[_-]?key)\s*([:=])\s*(['"]?)([^\s'",;}{]{8,})\3/gi,
    replace: (_m, key, sep, quote, val) => {
      if (val.startsWith('[REDACTED')) {
        return _m;
      }
      return `${key}${sep}${quote}${REDACTED_MARKER}${quote}`;
    },
  },
];

const SENSITIVE_OBJECT_KEYS = new Set([
  'apikey',
  'api_key',
  'secret',
  'secretkey',
  'secret_key',
  'accesstoken',
  'access_token',
  'refreshtoken',
  'refresh_token',
  'privatekey',
  'private_key',
  'clientsecret',
  'client_secret',
  'twitch_stream_key',
  'authorization',
  'password',
]);

function previewSecret(raw: string): string {
  if (raw.length <= 8) {
    return '***';
  }
  return `${raw.slice(0, 4)}...${raw.slice(-2)}`;
}

export function redactString(input: string, fieldPath = '$'): RedactStringResult {
  let current = input;
  const findings: SecretFinding[] = [];

  for (const rule of PATTERN_RULES) {
    rule.regex.lastIndex = 0;
    current = current.replace(rule.regex, (...args: unknown[]) => {
      const fullMatch = String(args[0] ?? '');
      const groups = args.slice(1, -2).map((g) => (g === undefined ? '' : String(g)));
      findings.push(
        Object.freeze({
          kind: rule.kind,
          path: fieldPath,
          preview: previewSecret(fullMatch),
        }),
      );
      return rule.replace(fullMatch, ...groups);
    });
  }

  return Object.freeze({
    redacted: current,
    findings: Object.freeze(findings),
    count: findings.length,
  });
}

export function redactValue<T>(value: T, currentPath = '$'): RedactValueResult<T> {
  const findings: SecretFinding[] = [];

  const visit = (node: unknown, nodePath: string, parentKey?: string): unknown => {
    if (typeof node === 'string') {
      if (parentKey && SENSITIVE_OBJECT_KEYS.has(parentKey.toLowerCase())) {
        if (node.length > 0 && !node.startsWith('[REDACTED')) {
          findings.push(
            Object.freeze({
              kind:
                parentKey.toLowerCase() === 'twitch_stream_key'
                  ? 'TWITCH_STREAM_KEY'
                  : 'GENERIC_SECRET_FIELD',
              path: nodePath,
              preview: previewSecret(node),
            }),
          );
          return `${REDACTED_MARKER}`;
        }
      }
      const res = redactString(node, nodePath);
      if (res.count > 0) {
        findings.push(...res.findings);
      }
      return res.redacted;
    }

    if (node instanceof Uint8Array) {
      const text = Buffer.from(node).toString('utf8');
      const res = redactString(text, nodePath);
      if (res.count === 0) {
        return node;
      }
      findings.push(...res.findings);
      return Buffer.from(res.redacted, 'utf8');
    }

    if (Array.isArray(node)) {
      let changed = false;
      const out = node.map((item, idx) => {
        const next = visit(item, `${nodePath}[${idx}]`);
        if (next !== item) {
          changed = true;
        }
        return next;
      });
      return changed ? out : node;
    }

    if (node !== null && typeof node === 'object') {
      const record = node as Record<string, unknown>;
      const out: Record<string, unknown> = {};
      let changed = false;
      for (const [k, v] of Object.entries(record)) {
        const next = visit(v, `${nodePath}.${k}`, k);
        if (next !== v) {
          changed = true;
        }
        out[k] = next;
      }
      return changed ? out : node;
    }

    return node;
  };

  const redacted = visit(value, currentPath) as T;
  return Object.freeze({
    redacted,
    findings: Object.freeze(findings),
    count: findings.length,
  });
}

export function scanForSecrets(value: unknown): readonly SecretFinding[] {
  return redactValue(value).findings;
}

export function assertNoSecrets(value: unknown): void {
  const findings = scanForSecrets(value);
  if (findings.length > 0) {
    const summary = findings.map((f) => `${f.kind}@${f.path}`).join(', ');
    throw new Error(`Secret leakage detected prior to CAS commit: ${summary}`);
  }
}
