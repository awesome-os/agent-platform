/**
 * @file scripts/ingest-verified-notebooks.test.mts
 * @description Hermetic unit & integration tests for `ingest-verified-notebooks.mts` (Issue #82).
 */

import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import {
  B8G_ABI_MAGIC_V3,
  DATASET_SCHEMA,
  computeRecordSha256,
  ingestVerifiedNotebooks,
  parseNotebookMarkdown,
  runCli,
  serializeJsonl,
  type VerifiedNotebookRecord,
} from './ingest-verified-notebooks.mts';

describe('scripts/ingest-verified-notebooks — CI-Verified Notebook Dataset Indexer', () => {
  it('parses tier1/tier2/tier3 cells paired with golden OUTPUT blocks and skips unverified cells', () => {
    const fixture = [
      '# Test Notebook',
      '',
      'Regular non-notebook code block should be ignored:',
      '```ts',
      'console.log("ignored")',
      '```',
      '',
      '```ts notebook="tier1" id="cell-t1" expect-exit="0"',
      'console.log("hello-t1")',
      '```',
      '',
      '<!-- OUTPUT:BEGIN id="cell-t1" -->',
      '```text',
      'hello-t1',
      '```',
      '<!-- OUTPUT:END -->',
      '',
      '```pwsh notebook="tier2" id="cell-t2"',
      'Write-Output "hello-t2"',
      '```',
      '',
      '<!-- OUTPUT:BEGIN id="cell-t2" -->',
      'hello-t2',
      '<!-- OUTPUT:END -->',
      '',
      '```cli notebook="tier3" id="cell-t3"',
      'echo "hello-t3"',
      '```',
      '',
      '<!-- OUTPUT:BEGIN id="cell-t3" -->',
      '```text',
      'hello-t3',
      '```',
      '<!-- OUTPUT:END -->',
      '',
      '```ts notebook="tier1" id="missing-golden"',
      'console.log("no golden output")',
      '```',
      '',
      '```ts notebook="tier2"',
      'console.log("missing id")',
      '```',
      '',
      '```ts notebook="tier1" id="unclosed-golden"',
      'console.log("unclosed")',
      '```',
      '',
      '<!-- OUTPUT:BEGIN id="unclosed-golden" -->',
      '```text',
      'never closed',
      '```',
    ].join('\r\n');

    const parsed = parseNotebookMarkdown(fixture, 'fixture.md');
    assert.equal(parsed.cellsFound, 6);
    assert.equal(parsed.skippedCells, 3);
    assert.equal(parsed.records.length, 3);

    assert.deepEqual(
      parsed.records.map((r) => ({
        id: r.id,
        tier: r.tier,
        lang: r.lang,
        command: r.command,
        goldenOutput: r.goldenOutput,
        schema: r.schema,
        abiMagic: r.abiMagic,
      })),
      [
        {
          id: 'cell-t1',
          tier: 'tier1',
          lang: 'ts',
          command: 'console.log("hello-t1")',
          goldenOutput: 'hello-t1',
          schema: DATASET_SCHEMA,
          abiMagic: B8G_ABI_MAGIC_V3,
        },
        {
          id: 'cell-t2',
          tier: 'tier2',
          lang: 'pwsh',
          command: 'Write-Output "hello-t2"',
          goldenOutput: 'hello-t2',
          schema: DATASET_SCHEMA,
          abiMagic: B8G_ABI_MAGIC_V3,
        },
        {
          id: 'cell-t3',
          tier: 'tier3',
          lang: 'cli',
          command: 'echo "hello-t3"',
          goldenOutput: 'hello-t3',
          schema: DATASET_SCHEMA,
          abiMagic: B8G_ABI_MAGIC_V3,
        },
      ],
    );
  });

  it('produces deterministic SHA-256 hashes sensitive to command and goldenOutput', () => {
    const base = {
      file: 'sub/test.md',
      id: 'smoke',
      tier: 'tier1' as const,
      lang: 'ts',
      command: 'console.log(1)',
      goldenOutput: '1',
    };
    const h1 = computeRecordSha256(base);
    const h2 = computeRecordSha256({ ...base });
    const hChangedCmd = computeRecordSha256({ ...base, command: 'console.log(2)' });
    const hChangedOut = computeRecordSha256({ ...base, goldenOutput: '2' });

    assert.match(h1, /^[0-9a-f]{64}$/);
    assert.equal(h1, h2);
    assert.notEqual(h1, hChangedCmd);
    assert.notEqual(h1, hChangedOut);
  });

  it('ingests a temp directory and emits valid JSONL via ingestVerifiedNotebooks and runCli', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'b8g-nb-ingest-'));
    try {
      const nbDir = path.join(tmpDir, 'notebooks');
      const nestedDir = path.join(nbDir, 'nested');
      fs.mkdirSync(nestedDir, { recursive: true });

      fs.writeFileSync(
        path.join(nbDir, 'a.md'),
        [
          '```ts notebook="tier1" id="alpha"',
          'console.log("alpha")',
          '```',
          '<!-- OUTPUT:BEGIN id="alpha" -->',
          '```text',
          'alpha',
          '```',
          '<!-- OUTPUT:END -->',
        ].join('\n'),
        'utf8',
      );

      fs.writeFileSync(
        path.join(nestedDir, 'b.md'),
        [
          '```cli notebook="tier2" id="beta"',
          'echo beta',
          '```',
          '<!-- OUTPUT:BEGIN id="beta" -->',
          '```text',
          'beta',
          '```',
          '<!-- OUTPUT:END -->',
          '```cli notebook="tier3" id="unverified"',
          'echo nope',
          '```',
        ].join('\n'),
        'utf8',
      );

      const outJsonl = path.join(tmpDir, 'out', 'dataset.jsonl');
      const summary = ingestVerifiedNotebooks({
        inputPath: nbDir,
        outPath: outJsonl,
      });

      assert.equal(summary.schema, DATASET_SCHEMA);
      assert.equal(summary.abiMagic, B8G_ABI_MAGIC_V3);
      assert.equal(summary.filesScanned, 2);
      assert.equal(summary.cellsFound, 3);
      assert.equal(summary.verifiedRecords, 2);
      assert.equal(summary.skippedCells, 1);

      const rawJsonl = fs.readFileSync(outJsonl, 'utf8');
      assert.equal(rawJsonl, serializeJsonl(summary.records));
      const lines = rawJsonl.trim().split('\n');
      assert.equal(lines.length, 2);
      const parsedLines = lines.map((line) => JSON.parse(line) as VerifiedNotebookRecord);
      assert.equal(parsedLines[0]?.file, 'a.md');
      assert.equal(parsedLines[1]?.file, 'nested/b.md');
      assert.equal(parsedLines[1]?.tier, 'tier2');

      const cliOutJsonl = path.join(tmpDir, 'out', 'cli.jsonl');
      assert.equal(runCli(['--help']), 0);
      assert.equal(runCli(['--in', nbDir, '--out', cliOutJsonl, '--json']), 0);
      assert.equal(fs.readFileSync(cliOutJsonl, 'utf8'), rawJsonl);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('ingests live workspace ../../notebooks/ when accessible', () => {
    const testDir = path.dirname(fileURLToPath(import.meta.url));
    const liveNotebooksDir = path.resolve(testDir, '../../../..', 'notebooks');
    if (!fs.existsSync(liveNotebooksDir)) {
      return;
    }

    const summary = ingestVerifiedNotebooks({ inputPath: liveNotebooksDir });
    assert.ok(summary.filesScanned >= 2);
    assert.ok(summary.verifiedRecords >= 8);
    assert.equal(summary.skippedCells, 0);
    for (const record of summary.records) {
      assert.equal(record.schema, DATASET_SCHEMA);
      assert.equal(record.abiMagic, B8G_ABI_MAGIC_V3);
      assert.match(record.sha256, /^[0-9a-f]{64}$/);
      assert.ok(record.command.length > 0);
      assert.ok(record.goldenOutput.length > 0);
    }
  });
});
