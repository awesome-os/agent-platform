#!/usr/bin/env node
/**
 * @file scripts/ingest-verified-notebooks.mts
 * @description CI-verified Markdown notebook dataset indexer for `brain-rag` / `zer0q`
 * zero-hallucination grounding feeds (Issue #82).
 *
 * Parses Markdown notebooks containing `notebook="tier1|tier2|tier3"` cells paired
 * with committed `<!-- OUTPUT:BEGIN id="..." --> ... <!-- OUTPUT:END -->` golden
 * blocks and emits `b8g.verified-notebook-dataset.v1` JSONL records.
 *
 * Strictly `erasableSyntaxOnly: true`, Node 22+ built-ins only, zero HTTP IPC.
 */

import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

export const DATASET_SCHEMA = 'b8g.verified-notebook-dataset.v1' as const;
export const B8G_ABI_MAGIC_V3 = '0x33564742' as const;

export type NotebookTier = 'tier1' | 'tier2' | 'tier3';

export interface VerifiedNotebookRecord {
  schema: typeof DATASET_SCHEMA;
  abiMagic: typeof B8G_ABI_MAGIC_V3;
  file: string;
  id: string;
  tier: NotebookTier;
  lang: string;
  command: string;
  goldenOutput: string;
  sha256: string;
}

export interface ParsedNotebookFile {
  cellsFound: number;
  skippedCells: number;
  records: VerifiedNotebookRecord[];
}

export interface IngestOptions {
  inputPath: string;
  outPath?: string;
  baseDir?: string;
}

export interface IngestSummary {
  schema: typeof DATASET_SCHEMA;
  abiMagic: typeof B8G_ABI_MAGIC_V3;
  inputPath: string;
  outPath: string | null;
  filesScanned: number;
  cellsFound: number;
  verifiedRecords: number;
  skippedCells: number;
  records: VerifiedNotebookRecord[];
}

interface ParsedCell {
  id: string | null;
  tier: NotebookTier;
  lang: string;
  command: string;
}

const IGNORED_DIRS = new Set([
  '.git',
  '.holo-ci-runtime',
  '.awesome-oci',
  'node_modules',
  'dist',
  'build',
]);

export function computeRecordSha256(input: {
  file: string;
  id: string;
  tier: NotebookTier;
  lang: string;
  command: string;
  goldenOutput: string;
}): string {
  const canonical = JSON.stringify([
    DATASET_SCHEMA,
    B8G_ABI_MAGIC_V3,
    input.file,
    input.id,
    input.tier,
    input.lang,
    input.command,
    input.goldenOutput,
  ]);
  return createHash('sha256').update(canonical, 'utf8').digest('hex');
}

function extractAttr(info: string, name: string): string | null {
  const doubleQuoted = new RegExp(`(?:^|\\s)${name}\\s*=\\s*"([^"]+)"`);
  const dqMatch = doubleQuoted.exec(info);
  if (dqMatch) {
    return dqMatch[1];
  }
  const singleQuoted = new RegExp(`(?:^|\\s)${name}\\s*=\\s*'([^']+)'`);
  const sqMatch = singleQuoted.exec(info);
  if (sqMatch) {
    return sqMatch[1];
  }
  return null;
}

function isNotebookTier(value: string | null): value is NotebookTier {
  return value === 'tier1' || value === 'tier2' || value === 'tier3';
}

function unwrapGoldenBlock(rawLines: string[]): string {
  const joined = rawLines.join('\n').trim();
  if (!joined) {
    return '';
  }
  const lines = joined.split('\n');
  const first = lines[0]?.trim() ?? '';
  const last = lines[lines.length - 1]?.trim() ?? '';
  const openFence = /^(`{3,})(.*)$/.exec(first);
  if (openFence && lines.length >= 2) {
    const fenceTicks = openFence[1];
    if (last === fenceTicks || (last.startsWith(fenceTicks) && /^`+$/.test(last))) {
      return lines.slice(1, -1).join('\n');
    }
  }
  return joined;
}

/**
 * Parse a single Markdown notebook string into verified dataset records.
 * Cells with `notebook="tier1|tier2|tier3"` that lack an `id` or a matching
 * `<!-- OUTPUT:BEGIN id="..." --> ... <!-- OUTPUT:END -->` block are skipped.
 */
export function parseNotebookMarkdown(
  markdown: string,
  fileLabel: string,
): ParsedNotebookFile {
  const lines = markdown.replace(/\r\n/g, '\n').split('\n');
  const cells: ParsedCell[] = [];
  const goldenOutputs = new Map<string, string>();

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    // Check for golden output block: <!-- OUTPUT:BEGIN id="..." -->
    const beginMatch = /^<!--\s*OUTPUT:BEGIN\s+([^>]+?)\s*-->$/.exec(line.trim());
    if (beginMatch) {
      const attrs = beginMatch[1];
      const outputId = extractAttr(attrs, 'id');
      const blockLines: string[] = [];
      i += 1;
      let closed = false;
      while (i < lines.length) {
        if (/^<!--\s*OUTPUT:END\s*-->$/.test(lines[i].trim())) {
          closed = true;
          break;
        }
        blockLines.push(lines[i]);
        i += 1;
      }
      if (closed && outputId && !goldenOutputs.has(outputId)) {
        goldenOutputs.set(outputId, unwrapGoldenBlock(blockLines));
      }
      i += 1;
      continue;
    }

    // Check for fenced code block: ```lang notebook="tier1" id="..."
    const fenceMatch = /^(`{3,})([^\r\n]*)$/.exec(line.trimStart());
    if (fenceMatch) {
      const fence = fenceMatch[1];
      const info = fenceMatch[2].trim();
      const tokens = info.split(/\s+/);
      const lang = tokens[0] ?? '';
      const tierAttr = extractAttr(info, 'notebook');
      const idAttr = extractAttr(info, 'id');

      const bodyLines: string[] = [];
      i += 1;
      while (i < lines.length) {
        const closing = lines[i].trim();
        if (closing.startsWith(fence) && /^`+$/.test(closing)) {
          break;
        }
        bodyLines.push(lines[i]);
        i += 1;
      }

      if (isNotebookTier(tierAttr)) {
        cells.push({
          id: idAttr,
          tier: tierAttr,
          lang: lang || 'text',
          command: bodyLines.join('\n').trim(),
        });
      }
      i += 1;
      continue;
    }

    i += 1;
  }

  const records: VerifiedNotebookRecord[] = [];
  let skippedCells = 0;

  for (const cell of cells) {
    if (!cell.id || !cell.command) {
      skippedCells += 1;
      continue;
    }
    const goldenOutput = goldenOutputs.get(cell.id);
    if (goldenOutput === undefined) {
      skippedCells += 1;
      continue;
    }
    const sha256 = computeRecordSha256({
      file: fileLabel,
      id: cell.id,
      tier: cell.tier,
      lang: cell.lang,
      command: cell.command,
      goldenOutput,
    });
    records.push({
      schema: DATASET_SCHEMA,
      abiMagic: B8G_ABI_MAGIC_V3,
      file: fileLabel,
      id: cell.id,
      tier: cell.tier,
      lang: cell.lang,
      command: cell.command,
      goldenOutput,
      sha256,
    });
  }

  return {
    cellsFound: cells.length,
    skippedCells,
    records,
  };
}

export function collectMarkdownFiles(inputPath: string): string[] {
  const absInput = path.resolve(inputPath);
  if (!fs.existsSync(absInput)) {
    throw new Error(`Input path does not exist: ${absInput}`);
  }
  const stat = fs.statSync(absInput);
  if (stat.isFile()) {
    return [absInput];
  }
  if (!stat.isDirectory()) {
    throw new Error(`Input path is neither a file nor a directory: ${absInput}`);
  }

  const files: string[] = [];
  const walk = (dir: string): void => {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      if (entry.isDirectory()) {
        if (!IGNORED_DIRS.has(entry.name)) {
          walk(path.join(dir, entry.name));
        }
      } else if (entry.isFile() && entry.name.toLowerCase().endsWith('.md')) {
        files.push(path.join(dir, entry.name));
      }
    }
  };

  walk(absInput);
  return files;
}

export function serializeJsonl(records: readonly VerifiedNotebookRecord[]): string {
  if (records.length === 0) {
    return '';
  }
  return `${records.map((record) => JSON.stringify(record)).join('\n')}\n`;
}

export function ingestVerifiedNotebooks(options: IngestOptions): IngestSummary {
  const absInput = path.resolve(options.inputPath);
  const inputStat = fs.statSync(absInput);
  const files = collectMarkdownFiles(absInput);

  const records: VerifiedNotebookRecord[] = [];
  let cellsFound = 0;
  let skippedCells = 0;

  for (const filePath of files) {
    let fileLabel: string;
    if (options.baseDir) {
      fileLabel = path.relative(path.resolve(options.baseDir), filePath).replace(/\\/g, '/');
    } else if (inputStat.isDirectory()) {
      fileLabel = path.relative(absInput, filePath).replace(/\\/g, '/');
    } else {
      fileLabel = path.basename(filePath);
    }

    const content = fs.readFileSync(filePath, 'utf8');
    const parsed = parseNotebookMarkdown(content, fileLabel);
    cellsFound += parsed.cellsFound;
    skippedCells += parsed.skippedCells;
    records.push(...parsed.records);
  }

  let resolvedOut: string | null = null;
  if (options.outPath) {
    resolvedOut = path.resolve(options.outPath);
    fs.mkdirSync(path.dirname(resolvedOut), { recursive: true });
    fs.writeFileSync(resolvedOut, serializeJsonl(records), 'utf8');
  }

  return {
    schema: DATASET_SCHEMA,
    abiMagic: B8G_ABI_MAGIC_V3,
    inputPath: absInput,
    outPath: resolvedOut,
    filesScanned: files.length,
    cellsFound,
    verifiedRecords: records.length,
    skippedCells,
    records,
  };
}

function printHelp(): void {
  const usage = [
    'Usage: node scripts/ingest-verified-notebooks.mts [options]',
    '',
    'Options:',
    '  --in <dir-or-file>   Markdown notebook file or directory (default: ../../notebooks)',
    '  --out <jsonl>        Write b8g.verified-notebook-dataset.v1 JSONL output file',
    '  --json               Print structured ingestion summary as JSON',
    '  --help, -h           Show this help message',
  ].join('\n');
  process.stdout.write(`${usage}\n`);
}

export function runCli(argv: readonly string[]): number {
  let inputPath: string | undefined;
  let outPath: string | undefined;
  let jsonOutput = false;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      printHelp();
      return 0;
    }
    if (arg === '--json') {
      jsonOutput = true;
      continue;
    }
    if (arg === '--in') {
      const value = argv[i + 1];
      if (!value) {
        process.stderr.write('Error: --in requires a path argument\n');
        return 1;
      }
      inputPath = value;
      i += 1;
      continue;
    }
    if (arg === '--out') {
      const value = argv[i + 1];
      if (!value) {
        process.stderr.write('Error: --out requires a file path argument\n');
        return 1;
      }
      outPath = value;
      i += 1;
      continue;
    }
    process.stderr.write(`Error: Unknown argument: ${arg}\n`);
    return 1;
  }

  const scriptDir = path.dirname(fileURLToPath(import.meta.url));
  const defaultInput = path.resolve(scriptDir, '../../../..', 'notebooks');
  const resolvedInput = inputPath ?? defaultInput;

  const summary = ingestVerifiedNotebooks({
    inputPath: resolvedInput,
    outPath,
  });

  if (jsonOutput) {
    process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
  } else if (outPath) {
    process.stdout.write(
      `[ingest-verified-notebooks] files=${summary.filesScanned} verified=${summary.verifiedRecords} skipped=${summary.skippedCells} out=${summary.outPath}\n`,
    );
  } else {
    process.stdout.write(serializeJsonl(summary.records));
  }

  return 0;
}

const isDirectExecution =
  process.argv[1] !== undefined &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isDirectExecution) {
  process.exitCode = runCli(process.argv.slice(2));
}
