/**
 * @file tutorials/05-algebraic-error-loop.mts
 * @description Tutorial 05: Bounded K-Pass Algebraic Error-Loop LSP (`VerifyLoop`)
 * demonstrating deterministic auto-fix + minimal structured `ErrorDelta[]` + LSP 3.17
 * diagnostics enabling small local models to converge in <= 2 passes.
 */

import assert from 'node:assert/strict';
import {
  getTargetAdapter,
  runBoundedErrorLoop,
  toLspDiagnostics,
  type VirtualFileMap,
} from '../packages/error-loop-lsp/index.mts';

async function main(): Promise<void> {
  console.log('=== Tutorial 05: Algebraic Error-Loop LSP (Bounded K-Pass Convergence) ===');

  const adapter = getTargetAdapter('web-html-js');
  assert.ok(adapter, 'Expected web-html-js adapter');

  const brokenFiles: VirtualFileMap = {
    'index.html': [
      '<!doctype html>',
      '<html>',
      '<body>',
      '  <script src="./missing-app.js"></script>',
      '  <script>const ok = 42;</script>',
      '</body>',
      '</html>',
    ].join('\n'),
  };

  const initialDeltas = adapter.check(brokenFiles);
  const lspMap = toLspDiagnostics(initialDeltas);
  console.log(`Initial deltas detected: ${initialDeltas.length}`);
  for (const d of initialDeltas) {
    console.log(`  - [${d.ruleId}] ${d.file}:${d.line}:${d.col} invariant=${d.invariant} (${d.actual})`);
  }
  assert.ok(initialDeltas.length >= 1, 'Expected missing script asset delta');
  assert.ok(Object.keys(lspMap).length >= 1, 'Expected LSP 3.17 diagnostic map');

  // Small local model repair function receiving ONLY compressed algebraic deltas
  const result = await runBoundedErrorLoop(
    brokenFiles,
    adapter,
    (deltas, files) => {
      const next = { ...files };
      for (const delta of deltas) {
        if (delta.invariant === 'R ⊆ M' || delta.ruleId.includes('ASSET') || delta.actual.includes('missing-app.js')) {
          next['missing-app.js'] = 'export const ready = true;\n';
        }
      }
      return next;
    },
    5,
  );

  console.log(`Loop Converged: ${result.converged} in ${result.passes} passes (remaining errors=${result.remainingDeltas.filter((d) => d.severity === 'error').length})`);
  assert.equal(result.converged, true);
  assert.ok(result.passes <= 3, 'Expected convergence within 3 passes');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
