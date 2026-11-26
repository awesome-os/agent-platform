#!/usr/bin/env node
/**
 * @file packages/zer0q-substrate/cli.mts
 * @description ZER0Q Autonomous Agent Runner CLI (`zer0q run|lookup|alias|replay|trigger|leases`).
 * Supports `--json`, `--help`, and `--cas <dir>`, usable directly via `node cli.mts`
 * or `.\bin\zer0q.ps1`.
 *
 * Strictly `erasableSyntaxOnly: true`, Node 22+ built-ins only.
 */

import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { DeadhandController } from './deadhand.mts';
import {
  type DeadhandTriggerKind,
  SubstrateCasStore,
  SubstrateKvStore,
  type Zer0qJobManifest,
  Zer0qRunner,
} from './index.mts';

export interface Zer0qCliIo {
  readonly stdout: (line: string) => void;
  readonly stderr: (line: string) => void;
}

interface ParsedCliArgs {
  readonly positional: readonly string[];
  readonly flags: ReadonlyMap<string, string | boolean>;
}

function parseCliArgs(argv: readonly string[]): ParsedCliArgs {
  const positional: string[] = [];
  const flags = new Map<string, string | boolean>();

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]!;
    if (arg === '-h' || arg === '--help') {
      flags.set('help', true);
      continue;
    }
    if (arg.startsWith('--')) {
      const eqIdx = arg.indexOf('=');
      if (eqIdx !== -1) {
        flags.set(arg.slice(2, eqIdx), arg.slice(eqIdx + 1));
      } else {
        const key = arg.slice(2);
        const next = argv[i + 1];
        if (next !== undefined && !next.startsWith('-')) {
          flags.set(key, next);
          i += 1;
        } else {
          flags.set(key, true);
        }
      }
      continue;
    }
    positional.push(arg);
  }

  return { positional, flags };
}

function getStringFlag(flags: ReadonlyMap<string, string | boolean>, name: string): string | undefined {
  const val = flags.get(name);
  return typeof val === 'string' ? val : undefined;
}

function hasBoolFlag(flags: ReadonlyMap<string, string | boolean>, name: string): boolean {
  const val = flags.get(name);
  return val === true || val === 'true';
}

export function formatZer0qHelp(): string {
  return `zer0q — ZER0Q SUBSTRATE CAS/KV & DEADHAND CLI

Usage:
  node cli.mts <command> [subcommand] [options]
  .\\bin\\zer0q.ps1 <command> [subcommand] [options]

Commands:
  run <manifest.json>                  Execute manifest (with SKIMMER <2ms CAS short-circuit)
  lookup <manifest.json>               Dry-run SKIMMER CAS lookup (0 tokens)
  alias set <name> <casHash>           Atomically set alias pointer
  alias get <name>                     Read alias pointer
  alias rollback <name>                Atomically rollback alias to previous CAS hash
  alias list                           List all alias pointers
  replay <runId|digest> [--model M]    Hydrate & optionally re-execute run from CAS
  trigger add <id> --kind K --ref R --manifest M   Register DEADHAND trigger (cron|webhook|cas-diff|git)
  trigger list                         List registered DEADHAND triggers
  trigger fire <id> [--payload JSON]   Fire trigger with KV idempotency deduplication
  leases [list|reap|acquire|heartbeat|release]     Inspect leases & reap expired orphans

Options:
  --cas <dir>       CAS & KV state directory (default: $ZER0Q_CAS_DIR or tmpdir/zer0q-cli-cas)
  --json            Emit machine-readable JSON output
  -h, --help        Show this help message
`;
}

function loadManifestInput(
  manifestPath: string | undefined,
  flags: ReadonlyMap<string, string | boolean>,
): Zer0qJobManifest {
  if (manifestPath && fs.existsSync(manifestPath)) {
    const raw = JSON.parse(fs.readFileSync(manifestPath, 'utf8')) as Zer0qJobManifest;
    return raw;
  }
  const prompt = getStringFlag(flags, 'prompt');
  if (prompt) {
    return {
      apiVersion: 'zer0q.awesome-os/v1',
      name: getStringFlag(flags, 'name') ?? 'inline-cli-job',
      ...(getStringFlag(flags, 'alias') ? { alias: getStringFlag(flags, 'alias') } : {}),
      inputs: {
        agentsMd: getStringFlag(flags, 'agents') ?? 'AGENTS.md',
        skills: [],
        workspace: getStringFlag(flags, 'workspace') ?? 'workspace:cli',
        prompt,
      },
      execution: {
        model: getStringFlag(flags, 'model') ?? 'b8g-xenon-1b',
        maxTokens: Number(getStringFlag(flags, 'max-tokens') ?? 25_000),
        maxWallMs: Number(getStringFlag(flags, 'max-wall-ms') ?? 30_000),
      },
    };
  }
  if (manifestPath) {
    throw new Error(`Manifest file not found: ${manifestPath}`);
  }
  throw new Error('Missing manifest path or --prompt');
}

function createControllerFromFlags(flags: ReadonlyMap<string, string | boolean>): {
  readonly controller: DeadhandController;
  readonly statePath: string;
} {
  const casRoot =
    getStringFlag(flags, 'cas') ??
    process.env.ZER0Q_CAS_DIR ??
    path.join(os.tmpdir(), 'zer0q-cli-cas');
  const cas = new SubstrateCasStore(casRoot);
  const kv = new SubstrateKvStore();
  const runner = new Zer0qRunner(cas, kv);
  const controller = new DeadhandController({ cas, kv, runner });
  const statePath = path.join(cas.storeRoot, 'deadhand-state.json');
  controller.loadStateFile(statePath);
  return { controller, statePath };
}

function emitResult(
  io: Zer0qCliIo,
  asJson: boolean,
  payload: unknown,
  humanLine: string,
): void {
  if (asJson) {
    io.stdout(JSON.stringify(payload, null, 2));
  } else {
    io.stdout(humanLine);
  }
}

export async function runZer0qCli(
  argv: readonly string[],
  io: Zer0qCliIo = {
    stdout: (line) => console.log(line),
    stderr: (line) => console.error(line),
  },
): Promise<number> {
  const { positional, flags } = parseCliArgs(argv);
  const asJson = hasBoolFlag(flags, 'json');
  const cmd = positional[0];

  if (hasBoolFlag(flags, 'help') || !cmd || cmd === 'help') {
    if (asJson) {
      io.stdout(
        JSON.stringify({
          ok: true,
          commands: ['run', 'lookup', 'alias', 'replay', 'trigger', 'leases'],
          help: formatZer0qHelp(),
        }),
      );
    } else {
      io.stdout(formatZer0qHelp());
    }
    return 0;
  }

  try {
    const { controller, statePath } = createControllerFromFlags(flags);

    if (cmd === 'run') {
      const manifestPath = positional[1] ?? getStringFlag(flags, 'manifest');
      const manifest = loadManifestInput(manifestPath, flags);
      const outputOverride = getStringFlag(flags, 'output');
      const result = await controller.runner.run(manifest, (ctx) => {
        ctx.recordTokens(120, 1);
        return {
          output: outputOverride ? JSON.parse(outputOverride) : { ok: true, prompt: manifest.inputs.prompt },
          tokensUsed: 120,
        };
      });
      controller.saveStateFile(statePath);
      emitResult(
        io,
        asJson,
        result,
        `[zer0q run] status=${result.status} cached=${result.cached} runId=${result.runId} tokens=${result.tokensUsed} latency=${result.latencyMs.toFixed(3)}ms digest=${result.outputDigest.slice(0, 16)}...`,
      );
      return 0;
    }

    if (cmd === 'lookup') {
      const manifestPath = positional[1] ?? getStringFlag(flags, 'manifest');
      const manifest = loadManifestInput(manifestPath, flags);
      const result = controller.runner.lookup(manifest);
      emitResult(
        io,
        asJson,
        result,
        `[zer0q lookup] hit=${result.hit} tokens=0 latency=${result.latencyMs.toFixed(3)}ms inputDigest=${result.inputDigest.slice(0, 16)}...`,
      );
      return 0;
    }

    if (cmd === 'alias') {
      const sub = positional[1] ?? 'list';
      if (sub === 'set') {
        const name = positional[2];
        const casHash = positional[3];
        if (!name || !casHash) {
          throw new Error('Usage: zer0q alias set <name> <casHash>');
        }
        const res = controller.setAlias(name, casHash);
        controller.saveStateFile(statePath);
        emitResult(io, asJson, res, `[zer0q alias set] ${name} -> ${res.currentHash}`);
        return res.ok ? 0 : 1;
      }
      if (sub === 'get') {
        const name = positional[2];
        if (!name) {
          throw new Error('Usage: zer0q alias get <name>');
        }
        const value = controller.getAlias(name) ?? null;
        emitResult(io, asJson, { name, casHash: value }, `[zer0q alias get] ${name} = ${value ?? '<unset>'}`);
        return value !== null ? 0 : 1;
      }
      if (sub === 'rollback') {
        const name = positional[2];
        if (!name) {
          throw new Error('Usage: zer0q alias rollback <name>');
        }
        const res = controller.rollbackAlias(name);
        controller.saveStateFile(statePath);
        emitResult(
          io,
          asJson,
          res,
          `[zer0q alias rollback] ${name} ok=${res.ok} current=${res.currentHash ?? '<none>'}`,
        );
        return res.ok ? 0 : 1;
      }
      if (sub === 'list') {
        const aliases = controller.listAliases();
        emitResult(io, asJson, { aliases }, JSON.stringify(aliases));
        return 0;
      }
      throw new Error(`Unknown alias subcommand "${sub}". Use set|get|rollback|list.`);
    }

    if (cmd === 'replay') {
      const target = positional[1];
      if (!target) {
        throw new Error('Usage: zer0q replay <runId|digest> [--model <overrideModel>]');
      }
      const overrideModel = getStringFlag(flags, 'model');
      const res = await controller.replay(
        target,
        overrideModel,
        overrideModel
          ? (snap) => ({
              replayedFrom: snap.runId,
              originalDigest: snap.originalDigest,
              modelOverride: overrideModel,
              originalOutput: snap.originalOutput,
            })
          : undefined,
      );
      emitResult(
        io,
        asJson,
        res,
        `[zer0q replay] runId=${res.hydrated.runId} verified=${res.hydrated.verified} reexecuted=${res.reexecuted} model=${res.modelUsed}`,
      );
      return 0;
    }

    if (cmd === 'trigger') {
      const sub = positional[1] ?? 'list';
      if (sub === 'add') {
        const id = positional[2];
        const kind = (getStringFlag(flags, 'kind') ?? 'cron') as DeadhandTriggerKind;
        const ref = getStringFlag(flags, 'ref') ?? '0 * * * *';
        const manifestPath = getStringFlag(flags, 'manifest');
        if (!id) {
          throw new Error('Usage: zer0q trigger add <id> --kind <cron|webhook|cas-diff|git> --ref <ref> --manifest <path>');
        }
        const manifest = loadManifestInput(manifestPath, flags);
        controller.registerTrigger({
          id,
          kind,
          scheduleOrRef: ref,
          manifest,
        });
        controller.saveStateFile(statePath);
        const added = { ok: true, id, kind, scheduleOrRef: ref };
        emitResult(io, asJson, added, `[zer0q trigger add] id=${id} kind=${kind} ref=${ref}`);
        return 0;
      }
      if (sub === 'list') {
        const triggers = controller.listTriggers();
        emitResult(io, asJson, { triggers }, `[zer0q trigger list] count=${triggers.length}`);
        return 0;
      }
      if (sub === 'fire') {
        const id = positional[2];
        if (!id) {
          throw new Error('Usage: zer0q trigger fire <id> [--payload <json>]');
        }
        const payloadRaw = getStringFlag(flags, 'payload');
        const payload = payloadRaw ? JSON.parse(payloadRaw) : { firedAt: 'default-tick' };
        const res = await controller.fireTrigger(id, payload, (ctx) => {
          ctx.recordTokens(50, 1);
          return { output: { triggered: id, payload }, tokensUsed: 50 };
        });
        controller.saveStateFile(statePath);
        emitResult(
          io,
          asJson,
          res,
          `[zer0q trigger fire] id=${id} deduplicated=${res.deduplicated} status=${res.runResult?.status ?? 'DEDUPLICATED'}`,
        );
        return 0;
      }
      throw new Error(`Unknown trigger subcommand "${sub}". Use add|list|fire.`);
    }

    if (cmd === 'leases') {
      const sub = positional[1] ?? 'list';
      const nowFlag = getStringFlag(flags, 'now');
      const nowMs = nowFlag !== undefined ? Number(nowFlag) : Date.now();
      if (sub === 'list') {
        const leases = controller.listActiveLeases(nowMs);
        emitResult(io, asJson, { leases }, `[zer0q leases] active=${leases.length}`);
        return 0;
      }
      if (sub === 'reap') {
        const reaped = controller.reapOrphanLeases(nowMs);
        controller.saveStateFile(statePath);
        emitResult(io, asJson, { reaped }, `[zer0q leases reap] reaped=${reaped.length}`);
        return 0;
      }
      if (sub === 'acquire') {
        const runId = positional[2];
        if (!runId) {
          throw new Error('Usage: zer0q leases acquire <runId> [--owner <owner>] [--ttl <ms>]');
        }
        const owner = getStringFlag(flags, 'owner') ?? `cli:${process.pid}`;
        const ttlMs = Number(getStringFlag(flags, 'ttl') ?? 30_000);
        const lease = controller.acquireLease(runId, { owner, ttlMs, nowMs });
        controller.saveStateFile(statePath);
        emitResult(io, asJson, { ok: lease !== null, lease }, `[zer0q leases acquire] runId=${runId} ok=${lease !== null}`);
        return lease ? 0 : 1;
      }
      if (sub === 'heartbeat') {
        const runId = positional[2];
        const owner = getStringFlag(flags, 'owner') ?? `cli:${process.pid}`;
        const ttlFlag = getStringFlag(flags, 'ttl');
        if (!runId) {
          throw new Error('Usage: zer0q leases heartbeat <runId> --owner <owner>');
        }
        const lease = controller.heartbeatLease(
          runId,
          owner,
          ttlFlag !== undefined ? Number(ttlFlag) : undefined,
          nowMs,
        );
        controller.saveStateFile(statePath);
        emitResult(io, asJson, { ok: lease !== null, lease }, `[zer0q leases heartbeat] runId=${runId} ok=${lease !== null}`);
        return lease ? 0 : 1;
      }
      if (sub === 'release') {
        const runId = positional[2];
        const owner = getStringFlag(flags, 'owner') ?? `cli:${process.pid}`;
        if (!runId) {
          throw new Error('Usage: zer0q leases release <runId> --owner <owner>');
        }
        const released = controller.releaseLease(runId, owner);
        controller.saveStateFile(statePath);
        emitResult(io, asJson, { ok: released, runId }, `[zer0q leases release] runId=${runId} ok=${released}`);
        return released ? 0 : 1;
      }
      throw new Error(`Unknown leases subcommand "${sub}". Use list|reap|acquire|heartbeat|release.`);
    }

    throw new Error(`Unknown zer0q command "${cmd}". Run with --help for usage.`);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (asJson) {
      io.stderr(JSON.stringify({ ok: false, error: message }));
    } else {
      io.stderr(`[zer0q error] ${message}`);
    }
    return 1;
  }
}

const isEntrypoint =
  process.argv[1] !== undefined &&
  path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));

if (isEntrypoint) {
  runZer0qCli(process.argv.slice(2)).then((code) => {
    process.exitCode = code;
  });
}
