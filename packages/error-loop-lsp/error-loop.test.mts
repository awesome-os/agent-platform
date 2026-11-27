import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  TARGET_ADAPTERS,
  applyDeterministicAutoFixes,
  getTargetAdapter,
  runBoundedErrorLoop,
  toLspDiagnostics,
  type ErrorDelta,
  type VirtualFileMap,
} from './index.mts';

describe('TARGET_ADAPTERS registry', () => {
  it('registers all 6 canonical target adapters', () => {
    assert.equal(TARGET_ADAPTERS.length, 6);
    const ids = TARGET_ADAPTERS.map((a) => a.id);
    assert.deepEqual(ids, [
      'web-html-js',
      'react-vite-ts',
      'node-typescript',
      'python-fastapi',
      'docker-oci-deploy',
      'cloudflare-static-edge',
    ]);
    assert.equal(getTargetAdapter('non-existent'), undefined);
    assert.equal(getTargetAdapter('web-html-js')?.id, 'web-html-js');
  });
});

describe('Adapter 1: web-html-js', () => {
  const adapter = getTargetAdapter('web-html-js')!;

  it('passes a valid HTML + inline JS bundle', () => {
    const files: VirtualFileMap = {
      'index.html': `<!doctype html>
<html>
  <head><link rel="stylesheet" href="app.css"></head>
  <body>
    <p>Closure: R ⊆ M</p>
    <script>const x = 1 + 2; console.log(x);</script>
  </body>
</html>`,
      'app.css': 'body { margin: 0; }',
    };
    assert.deepEqual(adapter.check(files), []);
  });

  it('detects inline script SyntaxError, missing asset closure, and B8G014 LaTeX leak', () => {
    const files: VirtualFileMap = {
      'index.html': `<!doctype html>
<html>
  <head><link rel="stylesheet" href="missing.css"></head>
  <body>
    <p>Invariant: $$R \\subseteq M$$</p>
    <script>
      function broken( { return 1; }
    </script>
  </body>
</html>`,
    };
    const deltas = adapter.check(files);
    const rules = new Set(deltas.map((d) => d.ruleId));
    assert.ok(rules.has('WEB_SCRIPT_SYNTAX'));
    assert.ok(rules.has('WEB_REF_CLOSURE'));
    assert.ok(rules.has('B8G014'));

    const latexDelta = deltas.find((d) => d.ruleId === 'B8G014');
    assert.ok(latexDelta?.autoFix);
    assert.equal(latexDelta.autoFix.replace, '<span class="math">R ⊆ M</span>');
  });
});

describe('Adapter 2: react-vite-ts', () => {
  const adapter = getTargetAdapter('react-vite-ts')!;

  it('passes a valid React + Vite + TS project', () => {
    const files: VirtualFileMap = {
      'package.json': JSON.stringify({ name: 'demo', scripts: { dev: 'vite', build: 'vite build' } }),
      'index.html': '<!doctype html><html><body><div id="root"></div></body></html>',
      'src/main.tsx': `import './index.css';\nimport { App } from './App';\nconsole.log(App);`,
      'src/App.tsx': 'export const App = () => null;',
      'src/index.css': 'body { color: black; }',
    };
    assert.deepEqual(adapter.check(files), []);
  });

  it('detects broken package.json, missing root div (with autoFix), and unresolved ./ import', () => {
    const files: VirtualFileMap = {
      'package.json': '{ invalid json',
      'index.html': '<!doctype html><html><body></body></html>',
      'src/main.tsx': `import { Missing } from './DoesNotExist';`,
    };
    const deltas = adapter.check(files);
    const rules = new Set(deltas.map((d) => d.ruleId));
    assert.ok(rules.has('REACT_PKG_JSON'));
    assert.ok(rules.has('REACT_ROOT_DIV'));
    assert.ok(rules.has('REACT_IMPORT_CLOSURE'));

    const rootDelta = deltas.find((d) => d.ruleId === 'REACT_ROOT_DIV');
    assert.ok(rootDelta?.autoFix);
    const { files: fixed } = applyDeterministicAutoFixes(files, [rootDelta]);
    assert.ok(fixed['index.html']?.includes('<div id="root"></div>'));
  });
});

describe('Adapter 3: node-typescript', () => {
  const adapter = getTargetAdapter('node-typescript')!;

  it('passes a valid ESM Node TypeScript package', () => {
    const files: VirtualFileMap = {
      'package.json': JSON.stringify({
        name: 'svc',
        type: 'module',
        main: './src/index.mts',
        scripts: { test: 'pnx tsx --test' },
      }),
      'src/index.mts': `import { helper } from './helper.mts';\nexport const v = helper();`,
      'src/helper.mts': 'export function helper() { return 1; }',
    };
    assert.deepEqual(adapter.check(files), []);
  });

  it('catches non-module type, missing main, unresolved import, and legacy npx (with autoFix)', () => {
    const files: VirtualFileMap = {
      'package.json': JSON.stringify(
        {
          name: 'svc',
          type: 'commonjs',
          main: './src/missing.mts',
          scripts: { lint: 'npx biome check .' },
        },
        null,
        2,
      ),
      'src/app.mts': `import { nope } from './nope.mts';`,
    };
    const deltas = adapter.check(files);
    const rules = new Set(deltas.map((d) => d.ruleId));
    assert.ok(rules.has('NODE_PKG_ESM'));
    assert.ok(rules.has('NODE_MAIN_MISSING'));
    assert.ok(rules.has('NODE_NO_NPX'));
    assert.ok(rules.has('NODE_IMPORT_CLOSURE'));

    const { files: fixed, applied } = applyDeterministicAutoFixes(files, deltas);
    assert.ok(applied.length >= 2);
    assert.ok(fixed['package.json']?.includes('"type": "module"'));
    assert.ok(fixed['package.json']?.includes('pnx biome check .'));
    assert.ok(!fixed['package.json']?.includes('npx '));
  });
});

describe('Adapter 4: python-fastapi', () => {
  const adapter = getTargetAdapter('python-fastapi')!;

  it('passes a valid FastAPI service', () => {
    const files: VirtualFileMap = {
      'main.py': `from fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass Item(BaseModel):\n    name: str\n`,
      'requirements.txt': 'fastapi==0.115.0\nuvicorn==0.30.0\npydantic==2.9.0\n',
    };
    assert.deepEqual(adapter.check(files), []);
  });

  it('detects missing FastAPI() instantiation, unbalanced parens, and undeclared dependencies', () => {
    const files: VirtualFileMap = {
      'main.py': `from fastapi import FastAPI\nimport uvicorn\n\ndef broken(:\n    return (1 + 2\n`,
      'requirements.txt': 'uvicorn==0.30.0\n',
    };
    const deltas = adapter.check(files);
    const rules = new Set(deltas.map((d) => d.ruleId));
    assert.ok(rules.has('PY_FASTAPI_APP'));
    assert.ok(rules.has('PY_UNBALANCED_DELIM'));
    assert.ok(rules.has('PY_MISSING_DEP'));
  });
});

describe('Adapter 5: docker-oci-deploy', () => {
  const adapter = getTargetAdapter('docker-oci-deploy')!;

  it('passes a hardened Dockerfile with valid COPY, USER, and EXPOSE', () => {
    const files: VirtualFileMap = {
      Dockerfile: `FROM node:22-alpine
WORKDIR /app
COPY package.json ./
COPY src ./src
USER node
EXPOSE 8080/tcp
CMD ["node", "src/server.mjs"]
`,
      'package.json': '{}',
      'src/server.mjs': 'console.log("ok");',
    };
    assert.deepEqual(adapter.check(files), []);
  });

  it('detects missing FROM, missing COPY source, missing non-root USER warning, and invalid EXPOSE', () => {
    const files: VirtualFileMap = {
      Dockerfile: `WORKDIR /app
COPY dist/bundle.js ./
EXPOSE notaport
`,
    };
    const deltas = adapter.check(files);
    const rules = new Set(deltas.map((d) => d.ruleId));
    assert.ok(rules.has('DOCKER_MISSING_FROM'));
    assert.ok(rules.has('DOCKER_COPY_CLOSURE'));
    assert.ok(rules.has('DOCKER_INVALID_EXPOSE'));
    const userWarn = deltas.find((d) => d.ruleId === 'DOCKER_NO_NONROOT_USER');
    assert.equal(userWarn?.severity, 'warning');
  });
});

describe('Adapter 6: cloudflare-static-edge', () => {
  const adapter = getTargetAdapter('cloudflare-static-edge')!;

  it('passes valid wrangler.toml and wrangler.jsonc projects', () => {
    const tomlFiles: VirtualFileMap = {
      'wrangler.toml': `name = "edge-app"
main = "src/worker.ts"
compatibility_date = "2026-01-15"
`,
      'src/worker.ts': 'export default { fetch() { return new Response("ok"); } };',
    };
    assert.deepEqual(adapter.check(tomlFiles), []);

    const jsoncFiles: VirtualFileMap = {
      'wrangler.jsonc': `{
  // Static assets edge config
  "name": "static-site",
  "compatibility_date": "2026-02-01",
  "assets": { "directory": "public" }
}`,
      'public/index.html': '<h1>Edge</h1>',
    };
    assert.deepEqual(adapter.check(jsoncFiles), []);
  });

  it('detects missing entrypoint, invalid compatibility_date, and hardcoded secret leaks', () => {
    const files: VirtualFileMap = {
      'wrangler.toml': `name = "leaky"
main = "src/missing.ts"
compatibility_date = "2026-13-99"
`,
      'src/secrets.ts': `export const k1 = "sk-live1234567890";
export const k2 = "ghp_abcdef1234567890"; // holo-ci-allow-secret
export const k3 = "AKIAIOSFODNN7EXAMPLE"; // holo-ci-allow-secret
`,
    };
    const deltas = adapter.check(files);
    const rules = new Set(deltas.map((d) => d.ruleId));
    assert.ok(rules.has('CF_ENTRY_CLOSURE'));
    assert.ok(rules.has('CF_COMPAT_DATE'));
    const secretDeltas = deltas.filter((d) => d.ruleId === 'CF_SECRET_LEAK');
    assert.equal(secretDeltas.length, 3);
  });
});

describe('toLspDiagnostics & Bounded Error Loop', () => {
  it('converts ErrorDelta[] into standard LSP 3.17 Diagnostics grouped by file URI', () => {
    const sampleDeltas: ErrorDelta[] = [
      {
        ruleId: 'B8G014',
        severity: 'error',
        file: 'index.html',
        line: 5,
        col: 10,
        invariant: 'NO_RAW_LATEX_LEAK',
        expected: 'Rendered HTML',
        actual: 'Raw LaTeX',
      },
      {
        ruleId: 'DOCKER_NO_NONROOT_USER',
        severity: 'warning',
        file: 'Dockerfile',
        line: 3,
        col: 1,
        invariant: 'OCI_NONROOT_USER',
        expected: 'USER node',
        actual: 'No USER directive',
      },
    ];

    const lsp = toLspDiagnostics(sampleDeltas);
    assert.ok(lsp['file:///index.html']);
    assert.equal(lsp['file:///index.html']![0]!.severity, 1);
    assert.equal(lsp['file:///index.html']![0]!.range.start.line, 4);
    assert.equal(lsp['file:///index.html']![0]!.range.start.character, 9);
    assert.equal(lsp['file:///Dockerfile']![0]!.severity, 2);
  });

  it('converges a broken web-html-js app in 2 passes with deterministic autoFix + minimal model repair', async () => {
    const adapter = getTargetAdapter('web-html-js')!;
    const brokenFiles: VirtualFileMap = {
      'index.html': `<!doctype html>
<html>
  <head><link rel="stylesheet" href="Style.css" /></head>
  <body>
    <p>Math: $$R \\subseteq M$$</p>
    <script>
      function greet( {
        return "hi";
      }
    </script>
  </body>
</html>`,
      'style.css': 'body { color: #111; }',
    };

    const receivedDeltaRuleIds: string[][] = [];

    // Simulates a small/dumb model that only receives compact ErrorDelta[]
    const smallModelRepair = (deltas: ErrorDelta[], files: VirtualFileMap): VirtualFileMap => {
      receivedDeltaRuleIds.push(deltas.map((d) => d.ruleId));
      const next = { ...files };
      for (const delta of deltas) {
        if (delta.ruleId === 'WEB_SCRIPT_SYNTAX' && next[delta.file]) {
          next[delta.file] = next[delta.file]!.replace('function greet( {', 'function greet() {');
        }
      }
      return next;
    };

    const result = await runBoundedErrorLoop(brokenFiles, adapter, smallModelRepair, 5);
    assert.equal(result.converged, true);
    assert.equal(result.passes, 2);
    assert.equal(result.remainingDeltas.length, 0);

    // Deterministic autoFixes should have already resolved B8G014 and WEB_REF_CLOSURE before invoking the model
    assert.deepEqual(receivedDeltaRuleIds, [['WEB_SCRIPT_SYNTAX']]);
    assert.equal(result.trace[0]!.appliedAutoFixes.length, 2);
    assert.equal(result.trace[0]!.invokedModelRepair, true);
    assert.equal(result.trace[1]!.errorCount, 0);
  });

  it('stops cleanly at maxPasses when errors cannot be repaired', async () => {
    const adapter = getTargetAdapter('web-html-js')!;
    const brokenFiles: VirtualFileMap = {
      'index.html': '<script>const = ;</script>',
    };
    const noopModel = (_deltas: ErrorDelta[], files: VirtualFileMap): VirtualFileMap => files;
    const result = await runBoundedErrorLoop(brokenFiles, adapter, noopModel, 3);
    assert.equal(result.converged, false);
    assert.equal(result.passes, 3);
    assert.ok(result.remainingDeltas.length > 0);
  });
});
