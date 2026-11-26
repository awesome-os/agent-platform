import * as fs from 'node:fs';
import * as path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import {
  TARGET_ADAPTERS,
  getTargetAdapter,
  runBoundedErrorLoop,
  toLspDiagnostics,
  type ErrorDelta,
  type VirtualFileMap,
} from './index.mts';

const IGNORED_DIRS = new Set([
  'node_modules',
  '.git',
  '.b8g-cas',
  'dist',
  'build',
  '.next',
  '__pycache__',
]);

const MAX_FILE_BYTES = 1024 * 1024;

function readDirectoryToVirtualFiles(rootDir: string): VirtualFileMap {
  const absRoot = path.resolve(rootDir);
  if (!fs.existsSync(absRoot) || !fs.statSync(absRoot).isDirectory()) {
    throw new Error(`Directory not found: ${absRoot}`);
  }

  const files: VirtualFileMap = {};

  function walk(currentDir: string): void {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        if (!IGNORED_DIRS.has(entry.name)) {
          walk(fullPath);
        }
      } else if (entry.isFile()) {
        const stat = fs.statSync(fullPath);
        if (stat.size > MAX_FILE_BYTES) {
          continue;
        }
        const buf = fs.readFileSync(fullPath);
        // Skip binary files containing null bytes
        if (buf.includes(0)) {
          continue;
        }
        const relPath = path.relative(absRoot, fullPath).replace(/\\/g, '/');
        files[relPath] = buf.toString('utf8');
      }
    }
  }

  walk(absRoot);
  return files;
}

function printUsage(): void {
  console.log(`b8g-error-loop-lsp CLI

Usage:
  cli.mts list-adapters
  cli.mts check <adapterId> <dir>
  cli.mts lsp-json <adapterId> <dir>
  cli.mts demo-loop
`);
}

function cmdListAdapters(): number {
  for (const adapter of TARGET_ADAPTERS) {
    console.log(`${adapter.id.padEnd(24)} ${adapter.name}`);
    console.log(`${''.padEnd(24)} ${adapter.description}`);
  }
  return 0;
}

function cmdCheck(adapterId: string, dir: string): number {
  const adapter = getTargetAdapter(adapterId);
  if (!adapter) {
    console.error(`Unknown adapter: "${adapterId}". Run "list-adapters" to see available adapters.`);
    return 1;
  }
  const files = readDirectoryToVirtualFiles(dir);
  const deltas = adapter.check(files);
  const errorCount = deltas.filter((d) => d.severity === 'error').length;
  const warningCount = deltas.filter((d) => d.severity === 'warning').length;

  console.log(
    JSON.stringify(
      {
        adapter: adapter.id,
        fileCount: Object.keys(files).length,
        errorCount,
        warningCount,
        deltas,
      },
      null,
      2,
    ),
  );
  return errorCount === 0 ? 0 : 1;
}

function cmdLspJson(adapterId: string, dir: string): number {
  const adapter = getTargetAdapter(adapterId);
  if (!adapter) {
    console.error(`Unknown adapter: "${adapterId}". Run "list-adapters" to see available adapters.`);
    return 1;
  }
  const files = readDirectoryToVirtualFiles(dir);
  const deltas = adapter.check(files);
  const errorCount = deltas.filter((d) => d.severity === 'error').length;
  const diagnostics = toLspDiagnostics(deltas);

  console.log(JSON.stringify(diagnostics, null, 2));
  return errorCount === 0 ? 0 : 1;
}

async function cmdDemoLoop(): Promise<number> {
  const adapter = getTargetAdapter('web-html-js');
  if (!adapter) {
    console.error('web-html-js adapter unavailable');
    return 1;
  }

  const brokenApp: VirtualFileMap = {
    'index.html': `<!doctype html>
<html>
  <head>
    <title>Demo App</title>
    <link rel="stylesheet" href="Styles.css" />
  </head>
  <body>
    <p>Closure invariant: $$R \\subseteq M$$</p>
    <script>
      const count = 42
      function render( {
        console.log(count);
      }
    </script>
  </body>
</html>`,
    'styles.css': 'body { font-family: sans-serif; }',
  };

  // Simulated compact model repair function receiving ONLY ErrorDelta[]
  const demoModelRepair = (deltas: ErrorDelta[], files: VirtualFileMap): VirtualFileMap => {
    const next = { ...files };
    for (const delta of deltas) {
      if (delta.ruleId === 'WEB_SCRIPT_SYNTAX' && next[delta.file]) {
        next[delta.file] = next[delta.file]!.replace('function render( {', 'function render() {');
      }
    }
    return next;
  };

  const result = await runBoundedErrorLoop(brokenApp, adapter, demoModelRepair, 5);
  console.log(
    JSON.stringify(
      {
        adapterId: result.adapterId,
        converged: result.converged,
        passes: result.passes,
        trace: result.trace.map((t) => ({
          pass: t.pass,
          errorCount: t.errorCount,
          warningCount: t.warningCount,
          appliedAutoFixes: t.appliedAutoFixes.map((f) => f.description),
          invokedModelRepair: t.invokedModelRepair,
        })),
        finalLspDiagnostics: toLspDiagnostics(result.remainingDeltas),
      },
      null,
      2,
    ),
  );
  return result.converged ? 0 : 1;
}

export async function runCli(argv: string[]): Promise<number> {
  const [command, arg1, arg2] = argv;
  if (!command || command === '--help' || command === '-h') {
    printUsage();
    return command ? 0 : 1;
  }

  switch (command) {
    case 'list-adapters':
      return cmdListAdapters();
    case 'check':
      if (!arg1 || !arg2) {
        console.error('Usage: cli.mts check <adapterId> <dir>');
        return 1;
      }
      return cmdCheck(arg1, arg2);
    case 'lsp-json':
      if (!arg1 || !arg2) {
        console.error('Usage: cli.mts lsp-json <adapterId> <dir>');
        return 1;
      }
      return cmdLspJson(arg1, arg2);
    case 'demo-loop':
      return await cmdDemoLoop();
    default:
      console.error(`Unknown command: "${command}"`);
      printUsage();
      return 1;
  }
}

const isMain =
  typeof process.argv[1] === 'string' &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isMain) {
  runCli(process.argv.slice(2))
    .then((code) => {
      process.exitCode = code;
    })
    .catch((err: unknown) => {
      console.error(err instanceof Error ? err.message : String(err));
      process.exitCode = 1;
    });
}
