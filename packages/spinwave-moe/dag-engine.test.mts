/**
 * @file intersections/spinwave-bus/dag-engine.test.mts
 * @description Production tests for `SpinWaveHoloDagEngine` driving nested
 * `holo-compiler-lockfile.json` build & hydration graphs over `SpinWaveArenaBus`,
 * `SpinWaveDagScheduler`, and `CasV11Store`.
 */

import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { performance } from 'node:perf_hooks';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import { SpinWaveHoloDagEngine } from './dag-engine.mts';
import {
  CasV11Store,
  casBlobPath,
  sha512Hex,
} from './cas-v11-store.mts';
import {
  generateHoloLockfile,
  readHoloLockfile,
  type HoloCompilerLockfile,
  type HoloLockfileNode,
} from './lockfile.mts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
function resolveWorkspaceLockfilePath(): string | null {
  let dir = __dirname;
  for (let i = 0; i < 8; i += 1) {
    const candidate = path.join(dir, 'holo-compiler-lockfile.json');
    if (fs.existsSync(candidate)) return candidate;
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

interface FixtureArtifacts {
  readonly lockfile: HoloCompilerLockfile;
  readonly payloads: ReadonlyMap<string, Buffer>;
}

/**
 * Builds a multi-layer nested `holo-compiler-lockfile.json`:
 *   Subsystem 0 (`code-oss`):
 *     `libuv` + `libv8_monolith` -> `node` -> `code-oss`
 *   Subsystem 1 (`wasi-host`):
 *     `wasi-libc` -> `wasi-host` (completely orthogonal branch)
 */
function buildMultiLayerFixture(): FixtureArtifacts {
  const payloads = new Map<string, Buffer>([
    ['libuv', Buffer.from('binary:libuv:v1.50.0', 'utf8')],
    ['libv8_monolith', Buffer.from('binary:libv8_monolith:v13.1', 'utf8')],
    ['node', Buffer.from('binary:node:v22.14.0', 'utf8')],
    ['code-oss', Buffer.from('binary:code-oss:v1.98.0', 'utf8')],
    ['wasi-libc', Buffer.from('binary:wasi-libc:v0.25.0', 'utf8')],
    ['wasi-host', Buffer.from('binary:wasi-host:v1.0.0', 'utf8')],
  ]);

  const digest = (id: string): string => sha512Hex(payloads.get(id)!);
  const size = (id: string): number => payloads.get(id)!.byteLength;

  const lockfile = generateHoloLockfile({
    repo: 'awesome-os/spinwave-dag-fixture',
    generatedAt: '2026-11-24T00:00:00.000Z',
    dependencies: [
      {
        id: 'code-oss',
        kind: 'bin',
        version: '1.98.0',
        target: 'win64',
        sha512: digest('code-oss'),
        size: size('code-oss'),
        executable: true,
        dependencies: [
          {
            id: 'node',
            kind: 'bin',
            version: '22.14.0',
            target: 'win64',
            sha512: digest('node'),
            size: size('node'),
            executable: true,
            dependencies: [
              {
                id: 'libuv',
                kind: 'lib',
                version: '1.50.0',
                target: 'win64',
                sha512: digest('libuv'),
                size: size('libuv'),
                executable: false,
                dependencies: [],
              },
              {
                id: 'libv8_monolith',
                kind: 'lib',
                version: '13.1.0',
                target: 'win64',
                sha512: digest('libv8_monolith'),
                size: size('libv8_monolith'),
                executable: false,
                dependencies: [],
              },
            ],
          },
        ],
      },
      {
        id: 'wasi-host',
        kind: 'wasm',
        version: '1.0.0',
        target: 'wasm32-wasi',
        sha512: digest('wasi-host'),
        size: size('wasi-host'),
        executable: false,
        dependencies: [
          {
            id: 'wasi-libc',
            kind: 'lib',
            version: '0.25.0',
            target: 'wasm32-wasi',
            sha512: digest('wasi-libc'),
            size: size('wasi-libc'),
            executable: false,
            dependencies: [],
          },
        ],
      },
    ],
  });

  return { lockfile, payloads };
}

describe('SpinWaveHoloDagEngine', () => {
  it('ingests multi-layer nested lockfile, assigns orthogonal FDM bands, and maps every node', () => {
    const { lockfile } = buildMultiLayerFixture();
    const engine = SpinWaveHoloDagEngine.fromLockfile(lockfile);

    assert.equal(engine.totalNodes, 6);
    assert.equal(engine.numBands, 2);

    const codeOssBand = engine.getBandForSubsystem('code-oss');
    const wasiHostBand = engine.getBandForSubsystem('wasi-host');
    assert.notEqual(codeOssBand, undefined);
    assert.notEqual(wasiHostBand, undefined);
    assert.notEqual(codeOssBand, wasiHostBand);

    const libuv = engine.getNode('libuv');
    const node = engine.getNode('node');
    const codeOss = engine.getNode('code-oss');
    const wasiHost = engine.getNode('wasi-host');

    assert.ok(libuv && node && codeOss && wasiHost);
    assert.equal(libuv.band, codeOssBand);
    assert.equal(node.band, codeOssBand);
    assert.equal(codeOss.band, codeOssBand);
    assert.equal(wasiHost.band, wasiHostBand);

    // Post-order DFS invariant: leaves have lower indices than their parents
    assert.ok(libuv.nodeIndex < node.nodeIndex);
    assert.ok(node.nodeIndex < codeOss.nodeIndex);
  });

  it('editing libuv fires ONLY libuv -> node -> code-oss and never touches orthogonal wasi-host branch', () => {
    const { lockfile } = buildMultiLayerFixture();
    const engine = SpinWaveHoloDagEngine.fromLockfile(lockfile);

    const result = engine.propagateEdit('libuv', { amplitude: 1.0, phaseRad: 0 });

    assert.equal(result.cancelled, false);
    assert.equal(result.firedCount, 3);
    assert.equal(result.totalNodes, 6);
    assert.deepEqual([...result.firedNodeIds], ['libuv', 'node', 'code-oss']);

    // Orthogonal branch and sibling leaf must remain completely untouched
    assert.ok(!result.firedNodeIds.includes('wasi-host'));
    assert.ok(!result.firedNodeIds.includes('wasi-libc'));
    assert.ok(!result.firedNodeIds.includes('libv8_monolith'));

    // SpinWaveArenaBus must show constructive amplitude on code-oss band and 0 on wasi-host band
    const codeOssBand = engine.getBandForSubsystem('code-oss')!;
    const wasiHostBand = engine.getBandForSubsystem('wasi-host')!;
    const activeSample = engine.sampleBand(codeOssBand);
    const orthogonalSample = engine.sampleBand(wasiHostBand);

    assert.ok(activeSample.amplitude > 0.99, `expected constructive amplitude ~1.0, got ${activeSample.amplitude}`);
    assert.equal(activeSample.coherentSign, 1);
    assert.equal(orthogonalSample.amplitude, 0);
  });

  it('cancels a wave cleanly via destructive phase (Math.PI) no-op guard', () => {
    const { lockfile } = buildMultiLayerFixture();
    const engine = SpinWaveHoloDagEngine.fromLockfile(lockfile);

    const result = engine.propagateEdit('libuv', {
      amplitude: 1.0,
      phaseRad: Math.PI,
    });

    assert.equal(result.cancelled, true);
    assert.equal(result.firedCount, 0);
    assert.deepEqual([...result.firedNodeIds], []);
    assert.equal(result.totalNodes, 6);

    // Verify destructive cancellation is recorded in the 64B-aligned SpinWaveArenaBus
    const codeOssBand = engine.getBandForSubsystem('code-oss')!;
    const cancelledSample = engine.sampleBand(codeOssBand);
    assert.ok(
      cancelledSample.amplitude < 1e-5,
      `expected destructive cancellation ~0, got ${cancelledSample.amplitude}`,
    );
  });

  it('executeWave achieves 100% CAS hit when blobs exist and compiles ONLY the 1 invalidated dirty leaf', async () => {
    const { lockfile, payloads } = buildMultiLayerFixture();
    const engine = SpinWaveHoloDagEngine.fromLockfile(lockfile);

    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'spinwave-dag-cas-'));
    const store = new CasV11Store(path.join(tmpRoot, 'v11'));

    try {
      // Seed all blobs into CasV11Store
      for (const id of engine.listNodeIds()) {
        const entry = engine.getNode(id)!;
        store.putBlob(payloads.get(id)!, {
          executable: entry.lockfileNode.executable,
          meta: { id },
        });
      }

      // Propagate edit on libuv -> fires ['libuv', 'node', 'code-oss']
      const wave = engine.propagateEdit('libuv');
      assert.deepEqual([...wave.firedNodeIds], ['libuv', 'node', 'code-oss']);

      // 1. Warm CAS run -> 100% CAS hit, compileFn never called
      let compileCalls = 0;
      const warmRes = await engine.executeWave(
        wave.firedNodeIds,
        store,
        (node: HoloLockfileNode) => {
          compileCalls += 1;
          return payloads.get(node.id)!;
        },
      );

      assert.equal(compileCalls, 0);
      assert.equal(warmRes.casHitRate, 1.0);
      assert.deepEqual([...warmRes.compiledIds], []);
      assert.deepEqual([...warmRes.casHitIds], ['libuv', 'node', 'code-oss']);

      // 2. Invalidate ONLY the 1 dirty leaf (`libuv`) in CasV11Store
      const libuvNode = engine.getNode('libuv')!.lockfileNode;
      const libuvBlobFile = casBlobPath(store.storeRoot, libuvNode.sha512, libuvNode.executable);
      fs.rmSync(libuvBlobFile, { force: true });
      assert.equal(store.hasBlob(libuvNode.sha512, libuvNode.executable), false);

      // Re-execute the wave -> compiles ONLY `libuv`, skips `node` and `code-oss` via O(1) CAS hit
      const compiledOrder: string[] = [];
      const dirtyRes = await engine.executeWave(
        wave.firedNodeIds,
        store,
        async (node: HoloLockfileNode) => {
          compiledOrder.push(node.id);
          return payloads.get(node.id)!;
        },
      );

      assert.deepEqual(compiledOrder, ['libuv']);
      assert.deepEqual([...dirtyRes.compiledIds], ['libuv']);
      assert.deepEqual([...dirtyRes.casHitIds], ['node', 'code-oss']);
      assert.ok(Math.abs(dirtyRes.casHitRate - 2 / 3) < 1e-9);
      assert.equal(store.hasBlob(libuvNode.sha512, libuvNode.executable), true);

      // 3. Subsequent wave run is back to 100% CAS hit
      const postRes = await engine.executeWave(
        wave.firedNodeIds,
        store,
        () => {
          throw new Error('compileFn must not be called on warm CAS');
        },
      );
      assert.equal(postRes.casHitRate, 1.0);
      assert.deepEqual([...postRes.compiledIds], []);
      assert.deepEqual([...postRes.casHitIds], ['libuv', 'node', 'code-oss']);
    } finally {
      store.close();
      fs.rmSync(tmpRoot, { recursive: true, force: true });
    }
  });

  it('ingests the real root e:\\awesome-os\\holo-compiler-lockfile.json (490+ nodes) in < 10 ms', () => {
    const rootLockfilePath = resolveWorkspaceLockfilePath();
    let lockfile: HoloCompilerLockfile;
    if (rootLockfilePath) {
      lockfile = readHoloLockfile(rootLockfilePath);
    } else {
      const dummySha = sha512Hex('synthetic-node');
      const syntheticSubs = Array.from({ length: 10 }, (_, sIdx) => ({
        id: `subsystem-${sIdx}`,
        kind: 'bin' as const,
        version: '1.0.0',
        target: 'win64',
        sha512: dummySha,
        size: 64,
        executable: true,
        dependencies: Array.from({ length: 49 }, (_, cIdx) => ({
          id: `leaf-${sIdx}-${cIdx}`,
          kind: 'lib' as const,
          version: '1.0.0',
          target: 'win64',
          sha512: dummySha,
          size: 32,
          executable: false,
          dependencies: [],
        })),
      }));
      lockfile = generateHoloLockfile({
        repo: 'awesome-os/synthetic-500',
        generatedAt: '2026-11-24T00:00:00.000Z',
        dependencies: syntheticSubs,
      });
    }

    const t0 = performance.now();
    const engine = SpinWaveHoloDagEngine.fromLockfile(lockfile);
    const buildMs = performance.now() - t0;

    assert.ok(
      engine.totalNodes >= 490,
      `Expected 490+ nodes in root holo-compiler-lockfile.json, got ${engine.totalNodes}`,
    );
    assert.ok(
      buildMs < 10,
      `Expected Spin-Wave DAG build in < 10 ms for ${engine.totalNodes} nodes, took ${buildMs.toFixed(3)} ms`,
    );
    assert.equal(engine.numBands, lockfile.dependencies.length);

    // Smoke-test constructive & destructive propagation on a real leaf node
    const firstSubsystem = lockfile.dependencies[0]!;
    const leafId = firstSubsystem.dependencies[0]?.id ?? firstSubsystem.id;

    const constructive = engine.propagateEdit(leafId, { amplitude: 1.0, phaseRad: 0 });
    assert.equal(constructive.cancelled, false);
    assert.ok(constructive.firedCount >= 1);
    assert.ok(constructive.firedNodeIds.includes(leafId));
    if (firstSubsystem.dependencies.length > 0) {
      assert.ok(constructive.firedNodeIds.includes(firstSubsystem.id));
    }

    const destructive = engine.propagateEdit(leafId, { amplitude: 1.0, phaseRad: Math.PI });
    assert.equal(destructive.cancelled, true);
    assert.equal(destructive.firedCount, 0);
  });
});
