import * as path from 'node:path';
import * as vm from 'node:vm';
import { defineSection, f, s, type ExecutionContext, type StreamFrame } from '../schema/index.mts';

export const B8G_ERROR_LOOP_CONTEXT: ExecutionContext = {
  boundary: 'in-process',
  display: 'headless',
  lifetime: 'oneshot',
  transport: 'os-pipe',
  capability: {
    tenantId: 'local-user',
    scopes: ['lsp.verify'],
    epoch: 1,
    privateData: false,
  },
  locality: 'local-arena',
};

export const B8gErrorLoopSection = defineSection({
  id: 'b8g/error-loop-lsp',
  version: '1.0.0',
  substrate: s.sabRing({ ringBytes: 65536 }),
  context: B8G_ERROR_LOOP_CONTEXT,
  fields: {
    adapterId: f.string().maxBytes(64),
    passesExecuted: f.int32(),
    maxPasses: f.int32().default(5),
    remainingErrors: f.int32(),
    converged: f.boolean(),
  },
});

export type ErrorSeverity = 'error' | 'warning';

export interface AutoFix {
  description: string;
  file: string;
  search: string;
  replace: string;
}

export interface ErrorDelta {
  ruleId: string;
  severity: ErrorSeverity;
  file: string;
  line: number;
  col: number;
  invariant: string;
  expected: string;
  actual: string;
  autoFix?: AutoFix;
}

export type VirtualFileMap = Record<string, string>;

export interface LspPosition {
  line: number;
  character: number;
}

export interface LspRange {
  start: LspPosition;
  end: LspPosition;
}

export interface LspDiagnostic {
  range: LspRange;
  severity: 1 | 2 | 3 | 4;
  code: string;
  source: string;
  message: string;
  data?: {
    invariant: string;
    expected: string;
    actual: string;
    autoFix?: AutoFix;
  };
}

export type LspDiagnosticMap = Record<string, LspDiagnostic[]>;

export interface TargetErrorAdapter {
  readonly id: string;
  readonly name: string;
  readonly description: string;
  check(files: VirtualFileMap): ErrorDelta[];
}

export interface LoopPassTrace {
  pass: number;
  errorCount: number;
  warningCount: number;
  deltas: ErrorDelta[];
  appliedAutoFixes: AutoFix[];
  invokedModelRepair: boolean;
}

export interface BoundedLoopResult {
  adapterId: string;
  converged: boolean;
  passes: number;
  maxPasses: number;
  finalFiles: VirtualFileMap;
  remainingDeltas: ErrorDelta[];
  trace: LoopPassTrace[];
}

export type ModelRepairFn = (
  deltas: ErrorDelta[],
  files: VirtualFileMap,
) => VirtualFileMap | Promise<VirtualFileMap>;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function normalizeFilePath(filePath: string): string {
  const slashes = filePath.replace(/\\/g, '/');
  const normalized = path.posix.normalize(slashes);
  return normalized.replace(/^\.\//, '').replace(/^\/+/, '');
}

function buildNormalizedFileIndex(files: VirtualFileMap): Map<string, { originalKey: string; content: string }> {
  const index = new Map<string, { originalKey: string; content: string }>();
  for (const [key, content] of Object.entries(files)) {
    index.set(normalizeFilePath(key), { originalKey: key, content });
  }
  return index;
}

function offsetToLineCol(content: string, offset: number): { line: number; col: number } {
  const clamped = Math.max(0, Math.min(offset, content.length));
  let line = 1;
  let lastNewline = -1;
  for (let i = 0; i < clamped; i += 1) {
    if (content.charCodeAt(i) === 10) {
      line += 1;
      lastNewline = i;
    }
  }
  return { line, col: clamped - lastNewline };
}

function stripQueryAndHash(ref: string): string {
  return ref.split('#')[0]!.split('?')[0]!.trim();
}

function isExternalReference(ref: string): boolean {
  const trimmed = ref.trim();
  if (!trimmed) return true;
  if (trimmed.startsWith('#') || trimmed.startsWith('//')) return true;
  return /^(?:https?:|data:|mailto:|tel:|javascript:|blob:|about:)/i.test(trimmed);
}

function resolveRelativeAsset(fromFile: string, rawRef: string): string {
  const cleaned = stripQueryAndHash(rawRef);
  if (cleaned.startsWith('/')) {
    return normalizeFilePath(cleaned);
  }
  const dir = path.posix.dirname(normalizeFilePath(fromFile));
  return normalizeFilePath(dir === '.' ? cleaned : path.posix.join(dir, cleaned));
}

function resolveRelativeModule(
  fromFile: string,
  specifier: string,
  fileIndex: Map<string, { originalKey: string; content: string }>,
  extensions: readonly string[],
): string | undefined {
  const base = resolveRelativeAsset(fromFile, specifier);
  if (fileIndex.has(base)) {
    return base;
  }
  for (const ext of extensions) {
    const candidate = `${base}${ext}`;
    if (fileIndex.has(candidate)) {
      return candidate;
    }
  }
  // Support ESM TypeScript imports written with .js/.mjs pointing to .ts/.tsx/.mts
  if (base.endsWith('.js')) {
    const stem = base.slice(0, -3);
    for (const ext of ['.ts', '.tsx', '.mts']) {
      if (fileIndex.has(`${stem}${ext}`)) {
        return `${stem}${ext}`;
      }
    }
  } else if (base.endsWith('.mjs')) {
    const stem = base.slice(0, -4);
    if (fileIndex.has(`${stem}.mts`)) {
      return `${stem}.mts`;
    }
  }
  return undefined;
}

function collectRelativeImports(content: string): Array<{ specifier: string; index: number }> {
  const results: Array<{ specifier: string; index: number }> = [];
  const patterns: RegExp[] = [
    /(?:import|export)\s+(?:[^'"]*?\s+from\s+)?['"](\.\.?\/[^'"]+)['"]/g,
    /import\s*\(\s*['"](\.\.?\/[^'"]+)['"]\s*\)/g,
  ];
  for (const pattern of patterns) {
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(content)) !== null) {
      if (match[1]) {
        results.push({ specifier: match[1], index: match.index });
      }
    }
  }
  return results;
}

function toFileUri(file: string): string {
  const norm = normalizeFilePath(file);
  return `file:///${norm}`;
}

function stripJsoncComments(raw: string): string {
  let out = '';
  let inString = false;
  let escaped = false;
  let i = 0;
  while (i < raw.length) {
    const ch = raw[i]!;
    const next = raw[i + 1];
    if (inString) {
      out += ch;
      if (escaped) {
        escaped = false;
      } else if (ch === '\\') {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
      i += 1;
      continue;
    }
    if (ch === '"') {
      inString = true;
      out += ch;
      i += 1;
      continue;
    }
    if (ch === '/' && next === '/') {
      i += 2;
      while (i < raw.length && raw[i] !== '\n') i += 1;
      continue;
    }
    if (ch === '/' && next === '*') {
      i += 2;
      while (i < raw.length && !(raw[i] === '*' && raw[i + 1] === '/')) {
        if (raw[i] === '\n') out += '\n';
        i += 1;
      }
      i += 2;
      continue;
    }
    out += ch;
    i += 1;
  }
  return out.replace(/,\s*([}\]])/g, '$1');
}

function replaceLatexSymbols(expr: string): string {
  return expr
    .replace(/\\subseteq\b/g, '⊆')
    .replace(/\\subset\b/g, '⊂')
    .replace(/\\in\b/g, '∈')
    .replace(/\\rightarrow\b/g, '→')
    .replace(/\\to\b/g, '→')
    .replace(/\\times\b/g, '×')
    .replace(/\\le\b/g, '≤')
    .replace(/\\ge\b/g, '≥')
    .replace(/\\neq\b/g, '≠')
    .replace(/\\alpha\b/g, 'α')
    .replace(/\\beta\b/g, 'β')
    .replace(/\\lambda\b/g, 'λ')
    .trim();
}

// ---------------------------------------------------------------------------
// LSP 3.17 Conversion
// ---------------------------------------------------------------------------

export function toLspDiagnostics(deltas: ErrorDelta[]): LspDiagnosticMap {
  const grouped: LspDiagnosticMap = {};
  for (const delta of deltas) {
    const uri = toFileUri(delta.file);
    const list = grouped[uri] ?? [];
    const line = Math.max(0, delta.line - 1);
    const character = Math.max(0, delta.col - 1);
    const diagnostic: LspDiagnostic = {
      range: {
        start: { line, character },
        end: { line, character: character + 1 },
      },
      severity: delta.severity === 'error' ? 1 : 2,
      code: delta.ruleId,
      source: 'b8g-error-loop-lsp',
      message: `[${delta.invariant}] Expected: ${delta.expected} | Actual: ${delta.actual}`,
      data: {
        invariant: delta.invariant,
        expected: delta.expected,
        actual: delta.actual,
        ...(delta.autoFix ? { autoFix: delta.autoFix } : {}),
      },
    };
    list.push(diagnostic);
    grouped[uri] = list;
  }
  return grouped;
}

// ---------------------------------------------------------------------------
// Adapter 1: web-html-js
// ---------------------------------------------------------------------------

const webHtmlJsAdapter: TargetErrorAdapter = {
  id: 'web-html-js',
  name: 'Web HTML + Inline JS',
  description:
    'Validates inline <script> syntax via node:vm, local href/src asset closure (R ⊆ M), and unrendered LaTeX leaks (B8G014).',
  check(files: VirtualFileMap): ErrorDelta[] {
    const deltas: ErrorDelta[] = [];
    const fileIndex = buildNormalizedFileIndex(files);

    for (const [normPath, entry] of fileIndex.entries()) {
      if (!normPath.endsWith('.html') && !normPath.endsWith('.htm')) {
        continue;
      }
      const { originalKey: file, content } = entry;

      // 1. Validate inline <script> blocks using vm.Script
      const scriptRe = /<script\b([^>]*)>([\s\S]*?)<\/script>/gi;
      let scriptMatch: RegExpExecArray | null;
      while ((scriptMatch = scriptRe.exec(content)) !== null) {
        const attrs = scriptMatch[1] ?? '';
        const body = scriptMatch[2] ?? '';
        if (/\bsrc\s*=/i.test(attrs)) {
          continue;
        }
        const typeMatch = /\btype\s*=\s*["']([^"']+)["']/i.exec(attrs);
        const scriptType = typeMatch?.[1]?.trim().toLowerCase() ?? 'text/javascript';
        if (
          scriptType !== 'text/javascript' &&
          scriptType !== 'application/javascript' &&
          scriptType !== 'module'
        ) {
          continue;
        }
        if (!body.trim()) {
          continue;
        }

        const bodyStartOffset = scriptMatch.index + scriptMatch[0].indexOf('>') + 1;
        const startPos = offsetToLineCol(content, bodyStartOffset);

        try {
          new vm.Script(body, { filename: file });
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : String(err);
          // For type="module", vm.Script cannot parse top-level import/export syntax directly
          if (
            scriptType === 'module' &&
            (message.includes('Cannot use import statement') ||
              message.includes("Unexpected token 'export'"))
          ) {
            continue;
          }
          let errLine = startPos.line;
          if (err instanceof Error && err.stack) {
            const stackLineMatch = /:(\d+)\r?\n/.exec(err.stack);
            if (stackLineMatch?.[1]) {
              const relativeLine = Number.parseInt(stackLineMatch[1], 10);
              if (Number.isFinite(relativeLine) && relativeLine > 0) {
                errLine = startPos.line + relativeLine - 1;
              }
            }
          }
          deltas.push({
            ruleId: 'WEB_SCRIPT_SYNTAX',
            severity: 'error',
            file,
            line: errLine,
            col: 1,
            invariant: 'E_SCRIPT_VALID_JS',
            expected: 'Valid JavaScript syntax inside inline <script> block',
            actual: `SyntaxError: ${message}`,
          });
        }
      }

      // 2. Check R ⊆ M closure on local href / src references
      const refRe = /\b(href|src)\s*=\s*["']([^"']+)["']/gi;
      let refMatch: RegExpExecArray | null;
      while ((refMatch = refRe.exec(content)) !== null) {
        const attr = refMatch[1]!;
        const rawRef = refMatch[2]!;
        if (isExternalReference(rawRef)) {
          continue;
        }
        const resolved = resolveRelativeAsset(normPath, rawRef);
        if (!fileIndex.has(resolved)) {
          const pos = offsetToLineCol(content, refMatch.index);
          const lowerTarget = resolved.toLowerCase();
          const caseMatches = [...fileIndex.keys()].filter((k) => k.toLowerCase() === lowerTarget);
          const autoFix: AutoFix | undefined =
            caseMatches.length === 1
              ? {
                  description: `Fix asset reference casing "${rawRef}" -> "${caseMatches[0]}"`,
                  file,
                  search: refMatch[0],
                  replace: `${attr}="${caseMatches[0]}"`,
                }
              : undefined;
          deltas.push({
            ruleId: 'WEB_REF_CLOSURE',
            severity: 'error',
            file,
            line: pos.line,
            col: pos.col,
            invariant: 'R ⊆ M',
            expected: `Referenced asset "${resolved}" exists in virtual file map`,
            actual: `Missing local asset referenced by ${attr}="${rawRef}"`,
            ...(autoFix ? { autoFix } : {}),
          });
        }
      }

      // 3. Check unrendered LaTeX leaks in HTML prose (B8G014)
      // Mask <script> and <style> regions with spaces to preserve offsets
      const proseOnly = content.replace(
        /<(script|style)\b[^>]*>[\s\S]*?<\/\1>/gi,
        (segment) => segment.replace(/[^\n]/g, ' '),
      );
      const latexRe = /\$\$([^$]+)\$\$|\\\((.+?)\\\)|\\\[(.+?)\\\]/g;
      let latexMatch: RegExpExecArray | null;
      while ((latexMatch = latexRe.exec(proseOnly)) !== null) {
        const rawLeak = latexMatch[0];
        const inner = (latexMatch[1] ?? latexMatch[2] ?? latexMatch[3] ?? '').trim();
        const pos = offsetToLineCol(content, latexMatch.index);
        const rendered = replaceLatexSymbols(inner);
        deltas.push({
          ruleId: 'B8G014',
          severity: 'error',
          file,
          line: pos.line,
          col: pos.col,
          invariant: 'NO_RAW_LATEX_LEAK',
          expected: 'Rendered HTML/Unicode prose without raw LaTeX delimiters',
          actual: `Unrendered LaTeX leak "${rawLeak}" in HTML prose`,
          autoFix: {
            description: `Replace raw LaTeX "${rawLeak}" with rendered HTML span`,
            file,
            search: rawLeak,
            replace: `<span class="math">${rendered}</span>`,
          },
        });
      }
    }

    return deltas;
  },
};

// ---------------------------------------------------------------------------
// Adapter 2: react-vite-ts
// ---------------------------------------------------------------------------

const REACT_EXTENSIONS = [
  '.ts',
  '.tsx',
  '.mts',
  '.js',
  '.jsx',
  '.mjs',
  '.css',
  '.json',
  '/index.ts',
  '/index.tsx',
  '/index.js',
  '/index.jsx',
] as const;

const reactViteTsAdapter: TargetErrorAdapter = {
  id: 'react-vite-ts',
  name: 'React + Vite + TypeScript',
  description:
    'Validates package.json + scripts, index.html <div id="root"> mount, and R ⊆ M relative import closure across TS/TSX/JS/JSX/CSS.',
  check(files: VirtualFileMap): ErrorDelta[] {
    const deltas: ErrorDelta[] = [];
    const fileIndex = buildNormalizedFileIndex(files);

    // 1. Validate package.json + scripts
    const pkgEntry = fileIndex.get('package.json');
    if (!pkgEntry) {
      deltas.push({
        ruleId: 'REACT_PKG_MISSING',
        severity: 'error',
        file: 'package.json',
        line: 1,
        col: 1,
        invariant: 'PKG_JSON_PRESENT',
        expected: 'package.json exists at project root',
        actual: 'package.json is missing',
      });
    } else {
      try {
        const pkg = JSON.parse(pkgEntry.content) as Record<string, unknown>;
        const scripts = pkg.scripts as Record<string, unknown> | undefined;
        if (!scripts || typeof scripts !== 'object' || (!scripts.dev && !scripts.build)) {
          deltas.push({
            ruleId: 'REACT_PKG_SCRIPTS',
            severity: 'error',
            file: pkgEntry.originalKey,
            line: 1,
            col: 1,
            invariant: 'VITE_SCRIPTS_DEFINED',
            expected: 'package.json defines "scripts.dev" or "scripts.build"',
            actual: 'Missing dev/build scripts in package.json',
          });
        }
      } catch (err: unknown) {
        deltas.push({
          ruleId: 'REACT_PKG_JSON',
          severity: 'error',
          file: pkgEntry.originalKey,
          line: 1,
          col: 1,
          invariant: 'PKG_JSON_VALID',
          expected: 'Valid JSON in package.json',
          actual: `JSON parse error: ${err instanceof Error ? err.message : String(err)}`,
        });
      }
    }

    // 2. Validate index.html has <div id="root">
    const htmlEntry = fileIndex.get('index.html');
    if (!htmlEntry) {
      deltas.push({
        ruleId: 'REACT_INDEX_MISSING',
        severity: 'error',
        file: 'index.html',
        line: 1,
        col: 1,
        invariant: 'VITE_INDEX_HTML_PRESENT',
        expected: 'index.html exists at project root',
        actual: 'index.html is missing',
      });
    } else if (!/<div\b[^>]*\bid\s*=\s*["']root["'][^>]*>/i.test(htmlEntry.content)) {
      const bodyMatch = /<body\b[^>]*>/i.exec(htmlEntry.content);
      const pos = bodyMatch ? offsetToLineCol(htmlEntry.content, bodyMatch.index) : { line: 1, col: 1 };
      const autoFix: AutoFix | undefined = bodyMatch
        ? {
            description: 'Insert <div id="root"></div> mount point inside <body>',
            file: htmlEntry.originalKey,
            search: bodyMatch[0],
            replace: `${bodyMatch[0]}\n    <div id="root"></div>`,
          }
        : undefined;
      deltas.push({
        ruleId: 'REACT_ROOT_DIV',
        severity: 'error',
        file: htmlEntry.originalKey,
        line: pos.line,
        col: pos.col,
        invariant: 'REACT_ROOT_MOUNT_PRESENT',
        expected: 'index.html contains <div id="root"></div>',
        actual: 'No <div id="root"> mount element found in index.html',
        ...(autoFix ? { autoFix } : {}),
      });
    }

    // 3. Validate R ⊆ M local ./ and ../ import closure
    for (const [normPath, entry] of fileIndex.entries()) {
      const isCode = /\.(?:ts|tsx|mts|js|jsx|mjs)$/i.test(normPath);
      const isCss = normPath.endsWith('.css');
      if (!isCode && !isCss) {
        continue;
      }

      if (isCode) {
        for (const imp of collectRelativeImports(entry.content)) {
          const resolved = resolveRelativeModule(normPath, imp.specifier, fileIndex, REACT_EXTENSIONS);
          if (!resolved) {
            const pos = offsetToLineCol(entry.content, imp.index);
            deltas.push({
              ruleId: 'REACT_IMPORT_CLOSURE',
              severity: 'error',
              file: entry.originalKey,
              line: pos.line,
              col: pos.col,
              invariant: 'R ⊆ M',
              expected: `Local import "${imp.specifier}" resolves to an existing file`,
              actual: `Unresolved relative import "${imp.specifier}" in ${entry.originalKey}`,
            });
          }
        }
      } else if (isCss) {
        const cssImportRe = /@import\s+(?:url\()?['"](\.\.?\/[^'"]+)['"]\)?/g;
        let match: RegExpExecArray | null;
        while ((match = cssImportRe.exec(entry.content)) !== null) {
          const spec = match[1]!;
          const resolved = resolveRelativeModule(normPath, spec, fileIndex, ['.css']);
          if (!resolved) {
            const pos = offsetToLineCol(entry.content, match.index);
            deltas.push({
              ruleId: 'REACT_IMPORT_CLOSURE',
              severity: 'error',
              file: entry.originalKey,
              line: pos.line,
              col: pos.col,
              invariant: 'R ⊆ M',
              expected: `CSS @import "${spec}" resolves to an existing stylesheet`,
              actual: `Unresolved CSS import "${spec}" in ${entry.originalKey}`,
            });
          }
        }
      }
    }

    return deltas;
  },
};

// ---------------------------------------------------------------------------
// Adapter 3: node-typescript
// ---------------------------------------------------------------------------

const NODE_TS_EXTENSIONS = [
  '.mts',
  '.ts',
  '.mjs',
  '.js',
  '.json',
  '/index.mts',
  '/index.ts',
  '/index.mjs',
  '/index.js',
] as const;

const nodeTypescriptAdapter: TargetErrorAdapter = {
  id: 'node-typescript',
  name: 'Node.js ESM TypeScript',
  description:
    'Validates package.json "type": "module" + main entrypoint, R ⊆ M relative import closure, and bans legacy npx commands.',
  check(files: VirtualFileMap): ErrorDelta[] {
    const deltas: ErrorDelta[] = [];
    const fileIndex = buildNormalizedFileIndex(files);

    const pkgEntry = fileIndex.get('package.json');
    if (!pkgEntry) {
      deltas.push({
        ruleId: 'NODE_PKG_MISSING',
        severity: 'error',
        file: 'package.json',
        line: 1,
        col: 1,
        invariant: 'PKG_JSON_PRESENT',
        expected: 'package.json exists at project root',
        actual: 'package.json is missing',
      });
    } else {
      try {
        const pkg = JSON.parse(pkgEntry.content) as Record<string, unknown>;

        // Check "type": "module"
        if (pkg.type !== 'module') {
          const typeFieldMatch = /"type"\s*:\s*"[^"]*"/.exec(pkgEntry.content);
          const autoFix: AutoFix = typeFieldMatch
            ? {
                description: 'Set package.json "type" to "module"',
                file: pkgEntry.originalKey,
                search: typeFieldMatch[0],
                replace: '"type": "module"',
              }
            : {
                description: 'Add "type": "module" to package.json',
                file: pkgEntry.originalKey,
                search: '{',
                replace: '{\n  "type": "module",',
              };
          deltas.push({
            ruleId: 'NODE_PKG_ESM',
            severity: 'error',
            file: pkgEntry.originalKey,
            line: typeFieldMatch ? offsetToLineCol(pkgEntry.content, typeFieldMatch.index).line : 1,
            col: 1,
            invariant: 'ESM_PACKAGE_TYPE',
            expected: 'package.json has "type": "module"',
            actual: `package.json "type" is ${JSON.stringify(pkg.type ?? null)}`,
            autoFix,
          });
        }

        // Check main entrypoint exists in files
        if (typeof pkg.main !== 'string' || !pkg.main.trim()) {
          deltas.push({
            ruleId: 'NODE_MAIN_MISSING',
            severity: 'error',
            file: pkgEntry.originalKey,
            line: 1,
            col: 1,
            invariant: 'R ⊆ M',
            expected: 'package.json defines a "main" entrypoint pointing to an existing file',
            actual: 'package.json "main" field is missing or empty',
          });
        } else {
          const mainPath = normalizeFilePath(pkg.main);
          if (!fileIndex.has(mainPath)) {
            deltas.push({
              ruleId: 'NODE_MAIN_MISSING',
              severity: 'error',
              file: pkgEntry.originalKey,
              line: 1,
              col: 1,
              invariant: 'R ⊆ M',
              expected: `Entrypoint "${mainPath}" exists in virtual file map`,
              actual: `package.json "main" target "${pkg.main}" does not exist`,
            });
          }
        }

        // Check legacy npx commands in scripts
        const scripts = pkg.scripts as Record<string, unknown> | undefined;
        if (scripts && typeof scripts === 'object') {
          for (const [scriptName, cmd] of Object.entries(scripts)) {
            if (typeof cmd === 'string' && /\bnpx\s+/.test(cmd)) {
              const idx = pkgEntry.content.indexOf(cmd);
              const pos = idx >= 0 ? offsetToLineCol(pkgEntry.content, idx) : { line: 1, col: 1 };
              deltas.push({
                ruleId: 'NODE_NO_NPX',
                severity: 'error',
                file: pkgEntry.originalKey,
                line: pos.line,
                col: pos.col,
                invariant: 'NO_LEGACY_NPX',
                expected: 'Scripts use "pnx " instead of legacy "npx "',
                actual: `Legacy "npx " command found in script "${scriptName}": ${cmd}`,
                autoFix: {
                  description: `Rewrite "npx " to "pnx " in script "${scriptName}"`,
                  file: pkgEntry.originalKey,
                  search: cmd,
                  replace: cmd.replace(/\bnpx\s+/g, 'pnx '),
                },
              });
            }
          }
        }
      } catch (err: unknown) {
        deltas.push({
          ruleId: 'NODE_PKG_JSON',
          severity: 'error',
          file: pkgEntry.originalKey,
          line: 1,
          col: 1,
          invariant: 'PKG_JSON_VALID',
          expected: 'Valid JSON in package.json',
          actual: `JSON parse error: ${err instanceof Error ? err.message : String(err)}`,
        });
      }
    }

    // Check R ⊆ M local relative imports
    for (const [normPath, entry] of fileIndex.entries()) {
      if (!/\.(?:ts|mts|js|mjs)$/i.test(normPath)) {
        continue;
      }
      for (const imp of collectRelativeImports(entry.content)) {
        const resolved = resolveRelativeModule(normPath, imp.specifier, fileIndex, NODE_TS_EXTENSIONS);
        if (!resolved) {
          const pos = offsetToLineCol(entry.content, imp.index);
          deltas.push({
            ruleId: 'NODE_IMPORT_CLOSURE',
            severity: 'error',
            file: entry.originalKey,
            line: pos.line,
            col: pos.col,
            invariant: 'R ⊆ M',
            expected: `Local import "${imp.specifier}" resolves to an existing file`,
            actual: `Unresolved relative import "${imp.specifier}" in ${entry.originalKey}`,
          });
        }
      }
    }

    return deltas;
  },
};

// ---------------------------------------------------------------------------
// Adapter 4: python-fastapi
// ---------------------------------------------------------------------------

const TRACKED_PY_DEPS = ['fastapi', 'uvicorn', 'pydantic', 'starlette', 'sqlalchemy', 'httpx'] as const;

function checkPythonBalancedDelimiters(file: string, content: string): ErrorDelta[] {
  const deltas: ErrorDelta[] = [];
  const stack: Array<{ ch: string; offset: number }> = [];
  const matching: Record<string, string> = { ')': '(', ']': '[', '}': '{' };

  let i = 0;
  while (i < content.length) {
    const ch = content[i]!;

    // Line comment
    if (ch === '#') {
      while (i < content.length && content[i] !== '\n') i += 1;
      continue;
    }

    // String literals (single, double, or triple quoted)
    if (ch === '"' || ch === "'") {
      const triple = content.slice(i, i + 3) === ch.repeat(3);
      const delim = triple ? ch.repeat(3) : ch;
      i += delim.length;
      while (i < content.length) {
        if (!triple && content[i] === '\\') {
          i += 2;
          continue;
        }
        if (content.startsWith(delim, i)) {
          i += delim.length;
          break;
        }
        i += 1;
      }
      continue;
    }

    if (ch === '(' || ch === '[' || ch === '{') {
      stack.push({ ch, offset: i });
    } else if (ch === ')' || ch === ']' || ch === '}') {
      const expectedOpen = matching[ch]!;
      const top = stack[stack.length - 1];
      if (!top || top.ch !== expectedOpen) {
        const pos = offsetToLineCol(content, i);
        deltas.push({
          ruleId: 'PY_UNBALANCED_DELIM',
          severity: 'error',
          file,
          line: pos.line,
          col: pos.col,
          invariant: 'BALANCED_SYNTAX_DELIMITERS',
          expected: top ? `Matching closing delimiter for "${top.ch}"` : 'Balanced delimiters',
          actual: `Unexpected closing "${ch}"`,
        });
      } else {
        stack.pop();
      }
    }
    i += 1;
  }

  for (const unclosed of stack) {
    const pos = offsetToLineCol(content, unclosed.offset);
    deltas.push({
      ruleId: 'PY_UNBALANCED_DELIM',
      severity: 'error',
      file,
      line: pos.line,
      col: pos.col,
      invariant: 'BALANCED_SYNTAX_DELIMITERS',
      expected: `Matching closing delimiter for "${unclosed.ch}"`,
      actual: `Unclosed "${unclosed.ch}"`,
    });
  }

  return deltas;
}

const pythonFastApiAdapter: TargetErrorAdapter = {
  id: 'python-fastapi',
  name: 'Python FastAPI Service',
  description:
    'Validates FastAPI() app instantiation, balanced brackets/parens, and requirements.txt / pyproject.toml dependency presence.',
  check(files: VirtualFileMap): ErrorDelta[] {
    const deltas: ErrorDelta[] = [];
    const fileIndex = buildNormalizedFileIndex(files);

    const pyEntries = [...fileIndex.entries()].filter(([normPath]) => normPath.endsWith('.py'));
    let hasFastApiApp = false;
    const importedDeps = new Map<string, { file: string; line: number; col: number }>();

    for (const [, entry] of pyEntries) {
      if (/\bFastAPI\s*\(/.test(entry.content)) {
        hasFastApiApp = true;
      }

      deltas.push(...checkPythonBalancedDelimiters(entry.originalKey, entry.content));

      const importRe = /^(?:from\s+([A-Za-z0-9_]+)|import\s+([A-Za-z0-9_]+))/gm;
      let match: RegExpExecArray | null;
      while ((match = importRe.exec(entry.content)) !== null) {
        const mod = (match[1] ?? match[2] ?? '').toLowerCase();
        if ((TRACKED_PY_DEPS as readonly string[]).includes(mod) && !importedDeps.has(mod)) {
          const pos = offsetToLineCol(entry.content, match.index);
          importedDeps.set(mod, { file: entry.originalKey, line: pos.line, col: pos.col });
        }
      }
    }

    if (!hasFastApiApp) {
      const targetFile = pyEntries[0]?.[1].originalKey ?? 'main.py';
      deltas.push({
        ruleId: 'PY_FASTAPI_APP',
        severity: 'error',
        file: targetFile,
        line: 1,
        col: 1,
        invariant: 'FASTAPI_APP_INSTANTIATED',
        expected: 'At least one Python module instantiates FastAPI()',
        actual: 'No FastAPI(...) app instantiation found',
      });
    }

    // Check dependency manifest
    const reqEntry = fileIndex.get('requirements.txt');
    const pyprojectEntry = fileIndex.get('pyproject.toml');

    if (importedDeps.size > 0 && !reqEntry && !pyprojectEntry) {
      deltas.push({
        ruleId: 'PY_DEP_MANIFEST',
        severity: 'error',
        file: 'requirements.txt',
        line: 1,
        col: 1,
        invariant: 'DEPS_DECLARED_IN_MANIFEST',
        expected: 'requirements.txt or pyproject.toml declares third-party dependencies',
        actual: `Missing requirements.txt and pyproject.toml for imports: ${[...importedDeps.keys()].join(', ')}`,
      });
    } else if (importedDeps.size > 0) {
      const manifestContent = `${reqEntry?.content ?? ''}\n${pyprojectEntry?.content ?? ''}`.toLowerCase();
      const manifestFile = reqEntry?.originalKey ?? pyprojectEntry?.originalKey ?? 'requirements.txt';
      for (const [dep] of importedDeps.entries()) {
        const depRe = new RegExp(`(?:^|[\\s"',=<>\\[\\(])${dep}(?:[\\s"',=<>\\[\\];~^!]|$)`, 'm');
        if (!depRe.test(manifestContent)) {
          deltas.push({
            ruleId: 'PY_MISSING_DEP',
            severity: 'error',
            file: manifestFile,
            line: 1,
            col: 1,
            invariant: 'DEPS_DECLARED_IN_MANIFEST',
            expected: `Dependency "${dep}" declared in ${manifestFile}`,
            actual: `Imported package "${dep}" is missing from ${manifestFile}`,
          });
        }
      }
    }

    return deltas;
  },
};

// ---------------------------------------------------------------------------
// Adapter 5: docker-oci-deploy
// ---------------------------------------------------------------------------

const dockerOciDeployAdapter: TargetErrorAdapter = {
  id: 'docker-oci-deploy',
  name: 'Docker / OCI Deployment',
  description:
    'Validates Dockerfile FROM base image, local COPY source closure (R ⊆ M), non-root USER warning, and valid EXPOSE integer ports.',
  check(files: VirtualFileMap): ErrorDelta[] {
    const deltas: ErrorDelta[] = [];
    const fileIndex = buildNormalizedFileIndex(files);

    const dockerfileEntries = [...fileIndex.entries()].filter(([normPath]) => {
      const base = path.posix.basename(normPath);
      return base === 'Dockerfile' || base.endsWith('.Dockerfile') || base.startsWith('Dockerfile.');
    });

    if (dockerfileEntries.length === 0) {
      deltas.push({
        ruleId: 'DOCKER_MISSING_FILE',
        severity: 'error',
        file: 'Dockerfile',
        line: 1,
        col: 1,
        invariant: 'DOCKERFILE_PRESENT',
        expected: 'Dockerfile exists in virtual file map',
        actual: 'No Dockerfile found',
      });
      return deltas;
    }

    const allNormPaths = [...fileIndex.keys()];

    for (const [, entry] of dockerfileEntries) {
      const { originalKey: file, content } = entry;
      const lines = content.split(/\r?\n/);
      let hasFrom = false;
      let hasNonRootUser = false;

      for (let idx = 0; idx < lines.length; idx += 1) {
        const rawLine = lines[idx]!;
        const trimmed = rawLine.trim();
        const lineNum = idx + 1;
        if (!trimmed || trimmed.startsWith('#')) {
          continue;
        }

        if (/^FROM\s+\S+/i.test(trimmed)) {
          hasFrom = true;
          continue;
        }

        if (/^USER\s+(\S+)/i.test(trimmed)) {
          const userMatch = /^USER\s+(\S+)/i.exec(trimmed);
          const userSpec = userMatch?.[1]?.toLowerCase() ?? '';
          if (userSpec && userSpec !== 'root' && userSpec !== '0' && !userSpec.startsWith('0:')) {
            hasNonRootUser = true;
          }
          continue;
        }

        if (/^EXPOSE\s+/i.test(trimmed)) {
          const tokens = trimmed.replace(/^EXPOSE\s+/i, '').split(/\s+/).filter(Boolean);
          for (const token of tokens) {
            const portPart = token.replace(/\/(?:tcp|udp)$/i, '');
            const portNum = Number(portPart);
            if (!/^\d+$/.test(portPart) || !Number.isInteger(portNum) || portNum < 1 || portNum > 65535) {
              deltas.push({
                ruleId: 'DOCKER_INVALID_EXPOSE',
                severity: 'error',
                file,
                line: lineNum,
                col: 1,
                invariant: 'VALID_TCP_UDP_PORT',
                expected: 'EXPOSE port is an integer between 1 and 65535',
                actual: `Invalid EXPOSE port "${token}"`,
              });
            }
          }
          continue;
        }

        if (/^COPY\s+/i.test(trimmed)) {
          const rest = trimmed.replace(/^COPY\s+/i, '').trim();
          const tokens = rest.split(/\s+/).filter(Boolean);
          let isMultiStage = false;
          const positional: string[] = [];
          for (const tok of tokens) {
            if (tok.startsWith('--from=')) {
              isMultiStage = true;
            } else if (tok.startsWith('--')) {
              continue;
            } else {
              positional.push(tok);
            }
          }
          if (isMultiStage || positional.length < 2) {
            continue;
          }
          const sources = positional.slice(0, -1);
          for (const src of sources) {
            if (src === '.' || src === './' || src === '*') {
              continue;
            }
            const cleanSrc = normalizeFilePath(src.replace(/\/+$/, ''));
            const existsAsFile = fileIndex.has(cleanSrc);
            const existsAsDir = allNormPaths.some((p) => p.startsWith(`${cleanSrc}/`));
            if (!existsAsFile && !existsAsDir) {
              deltas.push({
                ruleId: 'DOCKER_COPY_CLOSURE',
                severity: 'error',
                file,
                line: lineNum,
                col: 1,
                invariant: 'R ⊆ M',
                expected: `COPY source "${cleanSrc}" exists in virtual file map`,
                actual: `Missing COPY source "${src}" in ${file}`,
              });
            }
          }
        }
      }

      if (!hasFrom) {
        deltas.push({
          ruleId: 'DOCKER_MISSING_FROM',
          severity: 'error',
          file,
          line: 1,
          col: 1,
          invariant: 'DOCKER_BASE_IMAGE_REQUIRED',
          expected: 'Dockerfile declares at least one FROM base image',
          actual: `No FROM directive found in ${file}`,
        });
      }

      if (!hasNonRootUser) {
        deltas.push({
          ruleId: 'DOCKER_NO_NONROOT_USER',
          severity: 'warning',
          file,
          line: lines.length,
          col: 1,
          invariant: 'OCI_NONROOT_USER',
          expected: 'Dockerfile sets a non-root USER directive',
          actual: `No non-root USER directive found in ${file}`,
        });
      }
    }

    return deltas;
  },
};

// ---------------------------------------------------------------------------
// Adapter 6: cloudflare-static-edge
// ---------------------------------------------------------------------------

interface WranglerConfigSummary {
  main?: string;
  assetsDirectory?: string;
  compatibilityDate?: string;
}

function parseWranglerToml(content: string): WranglerConfigSummary {
  const summary: WranglerConfigSummary = {};
  let section = '';
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.replace(/#.*$/, '').trim();
    if (!line) continue;
    const secMatch = /^\[([^\]]+)\]$/.exec(line);
    if (secMatch?.[1]) {
      section = secMatch[1].trim();
      continue;
    }
    const kvMatch = /^([A-Za-z0-9_.-]+)\s*=\s*["']([^"']*)["']/.exec(line);
    if (!kvMatch) continue;
    const key = kvMatch[1]!;
    const value = kvMatch[2]!;
    if (!section && key === 'main') {
      summary.main = value;
    } else if (!section && key === 'compatibility_date') {
      summary.compatibilityDate = value;
    } else if (
      (section === 'assets' && key === 'directory') ||
      (!section && key === 'assets.directory')
    ) {
      summary.assetsDirectory = value;
    }
  }
  return summary;
}

function parseWranglerJsonc(content: string): WranglerConfigSummary {
  const parsed = JSON.parse(stripJsoncComments(content)) as Record<string, unknown>;
  const assets = parsed.assets as Record<string, unknown> | undefined;
  return {
    main: typeof parsed.main === 'string' ? parsed.main : undefined,
    assetsDirectory:
      assets && typeof assets.directory === 'string' ? assets.directory : undefined,
    compatibilityDate:
      typeof parsed.compatibility_date === 'string' ? parsed.compatibility_date : undefined,
  };
}

function isValidIsoDate(dateStr: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
    return false;
  }
  const [y, m, d] = dateStr.split('-').map((n) => Number.parseInt(n, 10));
  if (!y || !m || !d || m < 1 || m > 12 || d < 1 || d > 31) {
    return false;
  }
  const dt = new Date(Date.UTC(y, m - 1, d));
  return (
    dt.getUTCFullYear() === y &&
    dt.getUTCMonth() === m - 1 &&
    dt.getUTCDate() === d
  );
}

const cloudflareStaticEdgeAdapter: TargetErrorAdapter = {
  id: 'cloudflare-static-edge',
  name: 'Cloudflare Workers / Static Edge',
  description:
    'Validates wrangler.toml/jsonc entrypoints (main or assets.directory), YYYY-MM-DD compatibility_date, and enforces zero hardcoded secret leaks.',
  check(files: VirtualFileMap): ErrorDelta[] {
    const deltas: ErrorDelta[] = [];
    const fileIndex = buildNormalizedFileIndex(files);
    const allNormPaths = [...fileIndex.keys()];

    // 1. Locate and validate wrangler config
    const wranglerEntry =
      fileIndex.get('wrangler.toml') ??
      fileIndex.get('wrangler.jsonc') ??
      fileIndex.get('wrangler.json');

    if (!wranglerEntry) {
      deltas.push({
        ruleId: 'CF_WRANGLER_MISSING',
        severity: 'error',
        file: 'wrangler.toml',
        line: 1,
        col: 1,
        invariant: 'WRANGLER_CONFIG_PRESENT',
        expected: 'wrangler.toml or wrangler.jsonc exists at project root',
        actual: 'No wrangler.toml / wrangler.jsonc found',
      });
    } else {
      let summary: WranglerConfigSummary | undefined;
      try {
        summary = wranglerEntry.originalKey.endsWith('.toml')
          ? parseWranglerToml(wranglerEntry.content)
          : parseWranglerJsonc(wranglerEntry.content);
      } catch (err: unknown) {
        deltas.push({
          ruleId: 'CF_WRANGLER_PARSE',
          severity: 'error',
          file: wranglerEntry.originalKey,
          line: 1,
          col: 1,
          invariant: 'WRANGLER_CONFIG_VALID',
          expected: 'Valid wrangler configuration syntax',
          actual: `Parse error: ${err instanceof Error ? err.message : String(err)}`,
        });
      }

      if (summary) {
        // Entrypoint checks
        if (!summary.main && !summary.assetsDirectory) {
          deltas.push({
            ruleId: 'CF_ENTRY_MISSING',
            severity: 'error',
            file: wranglerEntry.originalKey,
            line: 1,
            col: 1,
            invariant: 'CF_ENTRYPOINT_DECLARED',
            expected: 'wrangler config declares "main" or "assets.directory"',
            actual: 'Neither "main" nor "assets.directory" is configured',
          });
        }

        if (summary.main) {
          const mainNorm = normalizeFilePath(summary.main);
          if (!fileIndex.has(mainNorm)) {
            deltas.push({
              ruleId: 'CF_ENTRY_CLOSURE',
              severity: 'error',
              file: wranglerEntry.originalKey,
              line: 1,
              col: 1,
              invariant: 'R ⊆ M',
              expected: `Worker entrypoint "${mainNorm}" exists in virtual file map`,
              actual: `Configured main entrypoint "${summary.main}" is missing`,
            });
          }
        }

        if (summary.assetsDirectory) {
          const dirNorm = normalizeFilePath(summary.assetsDirectory.replace(/\/+$/, ''));
          const hasAssets =
            dirNorm === '.' || dirNorm === ''
              ? allNormPaths.some((p) => p !== normalizeFilePath(wranglerEntry.originalKey))
              : allNormPaths.some((p) => p.startsWith(`${dirNorm}/`));
          if (!hasAssets) {
            deltas.push({
              ruleId: 'CF_ENTRY_CLOSURE',
              severity: 'error',
              file: wranglerEntry.originalKey,
              line: 1,
              col: 1,
              invariant: 'R ⊆ M',
              expected: `Static assets directory "${dirNorm}" contains at least one file`,
              actual: `Configured assets.directory "${summary.assetsDirectory}" has no files`,
            });
          }
        }

        // compatibility_date check
        if (!summary.compatibilityDate || !isValidIsoDate(summary.compatibilityDate)) {
          deltas.push({
            ruleId: 'CF_COMPAT_DATE',
            severity: 'error',
            file: wranglerEntry.originalKey,
            line: 1,
            col: 1,
            invariant: 'WRANGLER_COMPAT_DATE_ISO',
            expected: 'compatibility_date is a valid YYYY-MM-DD date',
            actual: `Invalid or missing compatibility_date: ${JSON.stringify(summary.compatibilityDate ?? null)}`,
          });
        }
      }
    }

    // 2. Zero hardcoded secret leak gate across all files
    const secretRe = /\b(sk-[A-Za-z0-9_-]{6,}|ghp_[A-Za-z0-9_]{6,}|AKIA[0-9A-Z]{8,})\b/g;
    for (const [, entry] of fileIndex.entries()) {
      let match: RegExpExecArray | null;
      while ((match = secretRe.exec(entry.content)) !== null) {
        const token = match[1]!;
        const prefix = token.startsWith('sk-')
          ? 'sk-'
          : token.startsWith('ghp_')
            ? 'ghp_'
            : 'AKIA';
        const pos = offsetToLineCol(entry.content, match.index);
        deltas.push({
          ruleId: 'CF_SECRET_LEAK',
          severity: 'error',
          file: entry.originalKey,
          line: pos.line,
          col: pos.col,
          invariant: 'ZERO_HARDCODED_SECRETS',
          expected: 'Zero hardcoded API keys or cloud credentials in source/config',
          actual: `Detected hardcoded secret token with prefix "${prefix}"`,
        });
      }
    }

    return deltas;
  },
};

// ---------------------------------------------------------------------------
// Adapter Registry
// ---------------------------------------------------------------------------

export const TARGET_ADAPTERS: readonly TargetErrorAdapter[] = [
  webHtmlJsAdapter,
  reactViteTsAdapter,
  nodeTypescriptAdapter,
  pythonFastApiAdapter,
  dockerOciDeployAdapter,
  cloudflareStaticEdgeAdapter,
] as const;

export function getTargetAdapter(id: string): TargetErrorAdapter | undefined {
  return TARGET_ADAPTERS.find((adapter) => adapter.id === id);
}

// ---------------------------------------------------------------------------
// Deterministic Auto-Fix & Bounded K-Pass Error Loop
// ---------------------------------------------------------------------------

export function applyDeterministicAutoFixes(
  files: VirtualFileMap,
  deltas: ErrorDelta[],
): { files: VirtualFileMap; applied: AutoFix[] } {
  const nextFiles: VirtualFileMap = { ...files };
  const applied: AutoFix[] = [];

  for (const delta of deltas) {
    const fix = delta.autoFix;
    if (!fix || fix.search === fix.replace) {
      continue;
    }
    const current = nextFiles[fix.file];
    if (typeof current !== 'string' || !current.includes(fix.search)) {
      continue;
    }
    nextFiles[fix.file] = current.replace(fix.search, fix.replace);
    applied.push(fix);
  }

  return { files: nextFiles, applied };
}

export async function runBoundedErrorLoop(
  initialFiles: VirtualFileMap,
  adapter: TargetErrorAdapter,
  modelRepairFn?: ModelRepairFn,
  maxPasses = 5,
): Promise<BoundedLoopResult> {
  const limit = Math.max(1, maxPasses);
  let currentFiles: VirtualFileMap = { ...initialFiles };
  const trace: LoopPassTrace[] = [];

  for (let pass = 1; pass <= limit; pass += 1) {
    const deltas = adapter.check(currentFiles);
    const errors = deltas.filter((d) => d.severity === 'error');
    const warnings = deltas.filter((d) => d.severity === 'warning');

    if (errors.length === 0) {
      trace.push({
        pass,
        errorCount: 0,
        warningCount: warnings.length,
        deltas,
        appliedAutoFixes: [],
        invokedModelRepair: false,
      });
      return {
        adapterId: adapter.id,
        converged: true,
        passes: pass,
        maxPasses: limit,
        finalFiles: currentFiles,
        remainingDeltas: deltas,
        trace,
      };
    }

    const { files: afterAutoFix, applied } = applyDeterministicAutoFixes(currentFiles, deltas);
    let invokedModelRepair = false;

    if (applied.length > 0) {
      currentFiles = afterAutoFix;
    }

    const postAutoFixDeltas = applied.length > 0 ? adapter.check(currentFiles) : deltas;
    const postAutoFixErrors = postAutoFixDeltas.filter((d) => d.severity === 'error');

    if (postAutoFixErrors.length > 0 && modelRepairFn) {
      const repaired = await modelRepairFn(postAutoFixDeltas, { ...currentFiles });
      currentFiles = { ...repaired };
      invokedModelRepair = true;
    }

    trace.push({
      pass,
      errorCount: errors.length,
      warningCount: warnings.length,
      deltas,
      appliedAutoFixes: applied,
      invokedModelRepair,
    });

    if (applied.length === 0 && !invokedModelRepair) {
      break;
    }
  }

  const remainingDeltas = adapter.check(currentFiles);
  const converged = remainingDeltas.filter((d) => d.severity === 'error').length === 0;

  return {
    adapterId: adapter.id,
    converged,
    passes: trace.length,
    maxPasses: limit,
    finalFiles: currentFiles,
    remainingDeltas,
    trace,
  };
}

// ---------------------------------------------------------------------------
// Pure Web Streams Transform Factory
// ---------------------------------------------------------------------------

export interface ErrorLoopTransformResult {
  readonly adapterId: string;
  readonly ok: boolean;
  readonly errorCount: number;
  readonly warningCount: number;
  readonly deltas: readonly ErrorDelta[];
  readonly diagnostics: LspDiagnosticMap;
}

/**
 * Creates a pure Web `TransformStream` that runs deterministic target verification
 * and LSP 3.17 diagnostic conversion per incoming source bundle frame.
 */
export function createErrorLoopTransform(
  adapterId: string,
): TransformStream<StreamFrame<Record<string, string>>, StreamFrame<ErrorLoopTransformResult>> {
  const adapter = getTargetAdapter(adapterId);
  if (!adapter) {
    throw new Error(`Unknown target error adapter: "${adapterId}"`);
  }

  const source = `b8g/error-loop-lsp:${adapter.id}`;

  return new TransformStream<StreamFrame<Record<string, string>>, StreamFrame<ErrorLoopTransformResult>>({
    transform(frame, controller) {
      const deltas = adapter.check(frame.payload);
      const diagnostics = toLspDiagnostics(deltas);
      let errorCount = 0;
      let warningCount = 0;
      for (const delta of deltas) {
        if (delta.severity === 'error') errorCount += 1;
        else warningCount += 1;
      }
      const ok = errorCount === 0;

      controller.enqueue({
        kind: ok ? 'data' : 'error',
        source,
        minPersona: ok ? frame.minPersona : 'devops',
        payload: {
          adapterId: adapter.id,
          ok,
          errorCount,
          warningCount,
          deltas,
          diagnostics,
        },
        ts: Date.now(),
        ...(frame.violations ? { violations: frame.violations } : {}),
      });
    },
  });
}
