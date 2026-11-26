/**
 * @file intersections/spinwave-bus/dag-engine.mts
 * @description Production Spin-Wave + Holo Lockfile DAG Engine (`SpinWaveHoloDagEngine`).
 *
 * Drives real `holo-compiler-lockfile.json` build & hydration graphs over:
 *   1. `SpinWaveDagScheduler` (64-byte-aligned `SharedArrayBuffer` damped-gravity wave DAG)
 *   2. `SpinWaveArenaBus` (lock-free seqlock FDM bands, 1 orthogonal band per top-level subsystem)
 *   3. `CasV11Store` (O(1) `v11/files/XX/<sha512>[-exec]` CAS hit gating)
 *
 * Strictly `erasableSyntaxOnly: true`, zero external dependencies, Node 22+ built-ins only.
 */

import { performance } from 'node:perf_hooks';
import { SpinWaveArenaBus, type SpinWaveSample } from './index.mts';
import {
  SpinWaveDagScheduler,
  type DagEdge,
  type NodePhasorSnapshot,
} from './scheduler.mts';
import { CasV11Store, type CasV11BlobKind } from './cas-v11-store.mts';
import {
  type HoloCompilerLockfile,
  type HoloLockfileNode,
  type HoloLockfileNodeKind,
} from './lockfile.mts';

// ============================================================================
// 1. Public Contracts
// ============================================================================

export interface SpinWaveHoloDagOptions {
  /** Global per-hop damping factor in (0, 1). Default: 0.85. */
  readonly alpha?: number;
  /** Constructive interference amplitude threshold to fire a node. Default: 0.25. */
  readonly threshold?: number;
  /** Gravity decay factor applied per epoch gap to stale priorities. Default: 0.85. */
  readonly gravityDecay?: number;
  /** Default Gilbert damping constant gamma for `SpinWaveArenaBus` bands. Default: 0.5. */
  readonly defaultDampingGamma?: number;
}

export interface DagNodeEntry {
  readonly nodeIndex: number;
  readonly lockfileNode: HoloLockfileNode;
  readonly band: number;
  readonly subsystemId: string;
}

export interface PropagateEditOptions {
  /** Wave amplitude injected at `dirtyNodeId`. Default: 1.0. */
  readonly amplitude?: number;
  /**
   * Wave phase in radians.
   * - `0` (default): constructive interference with reference carrier -> fires dependents.
   * - `Math.PI`: destructive antiphase cancellation / no-op guard -> `firedCount === 0`.
   */
  readonly phaseRad?: number;
}

export interface PropagateEditResult {
  readonly firedNodeIds: readonly string[];
  readonly cancelled: boolean;
  readonly waveDurationMs: number;
  readonly firedCount: number;
  readonly totalNodes: number;
}

export type WaveCompileFn = (
  node: HoloLockfileNode,
  entry: DagNodeEntry,
) => Uint8Array | string | Promise<Uint8Array | string>;

export interface WaveExecutionResult {
  readonly compiledIds: readonly string[];
  readonly casHitIds: readonly string[];
  readonly totalMs: number;
  readonly casHitRate: number;
}

// ============================================================================
// 2. Helpers
// ============================================================================

const CAS_BLOB_KINDS: ReadonlySet<string> = new Set<CasV11BlobKind>([
  'b8p',
  'npm',
  'cargo',
  'ghost',
  'object',
  'sidecar',
]);

function toCasBlobKind(kind: HoloLockfileNodeKind): CasV11BlobKind {
  if (CAS_BLOB_KINDS.has(kind)) {
    return kind as CasV11BlobKind;
  }
  return 'object';
}

// ============================================================================
// 3. SpinWaveHoloDagEngine
// ============================================================================

export class SpinWaveHoloDagEngine {
  readonly lockfile: HoloCompilerLockfile;
  readonly scheduler: SpinWaveDagScheduler;
  readonly bus: SpinWaveArenaBus;
  readonly totalNodes: number;
  readonly numBands: number;

  readonly #nodesById: ReadonlyMap<string, DagNodeEntry>;
  readonly #nodesByIndex: readonly DagNodeEntry[];
  readonly #bandBySubsystem: ReadonlyMap<string, number>;

  private constructor(
    lockfile: HoloCompilerLockfile,
    scheduler: SpinWaveDagScheduler,
    bus: SpinWaveArenaBus,
    nodesById: Map<string, DagNodeEntry>,
    nodesByIndex: DagNodeEntry[],
    bandBySubsystem: Map<string, number>,
  ) {
    this.lockfile = lockfile;
    this.scheduler = scheduler;
    this.bus = bus;
    this.totalNodes = nodesByIndex.length;
    this.numBands = bus.numBands;
    this.#nodesById = nodesById;
    this.#nodesByIndex = nodesByIndex;
    this.#bandBySubsystem = bandBySubsystem;
  }

  /**
   * Ingests any nested `holo-compiler-lockfile.json` tree into a
   * `SpinWaveDagScheduler` (directed edges from leaves -> parents) and allocates
   * a matching 64-byte-aligned `SpinWaveArenaBus` with 1 orthogonal FDM band per
   * top-level subsystem.
   *
   * Post-order DFS indexing guarantees that for every dependency edge
   * `child -> parent`, `child.nodeIndex < parent.nodeIndex`, so sorting any
   * subset of fired nodes by `nodeIndex` ascending yields strict bottom-up
   * topological order in O(k log k).
   */
  static fromLockfile(
    lockfile: HoloCompilerLockfile,
    opts?: SpinWaveHoloDagOptions,
  ): SpinWaveHoloDagEngine {
    const topDeps = lockfile.dependencies;
    if (!Array.isArray(topDeps) || topDeps.length === 0) {
      throw new RangeError('Cannot build SpinWaveHoloDagEngine from an empty lockfile');
    }

    const nodesById = new Map<string, DagNodeEntry>();
    const nodesByIndex: DagNodeEntry[] = [];
    const bandBySubsystem = new Map<string, number>();
    const edges: DagEdge[] = [];
    const edgeKeys = new Set<string>();
    const visiting = new Set<string>();

    const visit = (
      node: HoloLockfileNode,
      band: number,
      subsystemId: string,
    ): number => {
      const existing = nodesById.get(node.id);
      if (existing !== undefined) {
        return existing.nodeIndex;
      }

      if (visiting.has(node.id)) {
        throw new Error(`Dependency cycle detected while ingesting lockfile at "${node.id}"`);
      }
      visiting.add(node.id);

      const childIndices: number[] = [];
      const deps = node.dependencies;
      if (deps && deps.length > 0) {
        for (let i = 0; i < deps.length; i += 1) {
          childIndices.push(visit(deps[i]!, band, subsystemId));
        }
      }

      visiting.delete(node.id);

      const nodeIndex = nodesByIndex.length;
      const entry: DagNodeEntry = Object.freeze({
        nodeIndex,
        lockfileNode: node,
        band,
        subsystemId,
      });
      nodesById.set(node.id, entry);
      nodesByIndex.push(entry);

      // Direct spin-wave propagation edges: leaf/child -> parent
      for (let i = 0; i < childIndices.length; i += 1) {
        const childIndex = childIndices[i]!;
        if (childIndex === nodeIndex) {
          continue;
        }
        const key = `${childIndex}->${nodeIndex}`;
        if (!edgeKeys.has(key)) {
          edgeKeys.add(key);
          edges.push({ from: childIndex, to: nodeIndex });
        }
      }

      return nodeIndex;
    };

    for (let band = 0; band < topDeps.length; band += 1) {
      const topNode = topDeps[band]!;
      if (!bandBySubsystem.has(topNode.id)) {
        bandBySubsystem.set(topNode.id, band);
      }
      visit(topNode, band, topNode.id);
    }

    const scheduler = new SpinWaveDagScheduler({
      nodeCount: nodesByIndex.length,
      edges,
      alpha: opts?.alpha ?? 0.85,
      threshold: opts?.threshold ?? 0.25,
      gravityDecay: opts?.gravityDecay ?? 0.85,
    });

    const bus = new SpinWaveArenaBus({
      numBands: Math.max(1, topDeps.length),
      defaultDampingGamma: opts?.defaultDampingGamma ?? 0.5,
    });

    return new SpinWaveHoloDagEngine(
      lockfile,
      scheduler,
      bus,
      nodesById,
      nodesByIndex,
      bandBySubsystem,
    );
  }

  /** Looks up the `(nodeIndex, lockfileNode, band)` mapping for a node ID. */
  getNode(nodeId: string): DagNodeEntry | undefined {
    return this.#nodesById.get(nodeId);
  }

  /** Looks up the `(nodeIndex, lockfileNode, band)` mapping by dense scheduler index. */
  getNodeByIndex(nodeIndex: number): DagNodeEntry | undefined {
    return this.#nodesByIndex[nodeIndex];
  }

  /** Returns the orthogonal FDM band assigned to a top-level subsystem ID. */
  getBandForSubsystem(subsystemId: string): number | undefined {
    return this.#bandBySubsystem.get(subsystemId);
  }

  /** Returns all indexed node IDs in bottom-up topological order. */
  listNodeIds(): readonly string[] {
    return this.#nodesByIndex.map((entry) => entry.lockfileNode.id);
  }

  /** Samples the current spin-wave interference state on a subsystem's FDM band. */
  sampleBand(band: number): SpinWaveSample {
    return this.bus.sampleIntersection(band);
  }

  /** Reads a node's 64-byte-aligned phasor snapshot from the scheduler. */
  inspectNode(nodeId: string): NodePhasorSnapshot {
    const entry = this.#nodesById.get(nodeId);
    if (entry === undefined) {
      throw new Error(`Unknown lockfile node ID: "${nodeId}"`);
    }
    return this.scheduler.inspectNode(entry.nodeIndex);
  }

  /**
   * Injects a spin wave at `dirtyNodeId` against the coherent reference carrier:
   * - `phaseRad = 0`: constructive interference `(A/2, 0) + (A/2, 0) = (A, 0)` -> fires dependents.
   * - `phaseRad = Math.PI`: destructive antiphase cancellation `(A/2, 0) + (A/2, pi) = 0` -> no-op guard.
   *
   * Runs damped-gravity wave propagation on `SpinWaveDagScheduler`, records
   * interference into `SpinWaveArenaBus`, and returns the fired nodes in strict
   * bottom-up topological order.
   */
  propagateEdit(
    dirtyNodeId: string,
    opts?: PropagateEditOptions,
  ): PropagateEditResult {
    const entry = this.#nodesById.get(dirtyNodeId);
    if (entry === undefined) {
      throw new Error(`Unknown lockfile node ID: "${dirtyNodeId}"`);
    }

    const amplitude = opts?.amplitude ?? 1.0;
    const phaseRad = opts?.phaseRad ?? 0.0;
    if (!Number.isFinite(amplitude) || amplitude < 0) {
      throw new RangeError(`amplitude must be a finite non-negative number, got ${amplitude}`);
    }
    if (!Number.isFinite(phaseRad)) {
      throw new RangeError(`phaseRad must be a finite number, got ${phaseRad}`);
    }

    const startMs = performance.now();
    const halfAmp = amplitude * 0.5;

    // 1. Record coherent reference + edit wave superposition on the subsystem's FDM band
    this.bus.clearBand(entry.band);
    this.bus.emitWave(entry.band, halfAmp, 0.0);
    this.bus.emitWave(entry.band, halfAmp, phaseRad);

    // 2. Superpose reference carrier + edit wave in the 64B-aligned DAG scheduler
    this.scheduler.beginEpoch();
    this.scheduler.inject(entry.nodeIndex, halfAmp, 0.0);
    this.scheduler.inject(entry.nodeIndex, halfAmp, phaseRad);

    // 3. Propagate damped spin-wave front along leaf -> parent edges
    const firedCount = this.scheduler.propagateCount();
    const firedView = this.scheduler.getFiredView(firedCount);

    // Copy & sort ascending by post-order nodeIndex -> strict bottom-up topological order
    const sortedIndices = new Int32Array(firedCount);
    for (let i = 0; i < firedCount; i += 1) {
      sortedIndices[i] = firedView[i]!;
    }
    sortedIndices.sort();

    const firedNodeIds: string[] = new Array(firedCount);
    const touchedOtherBands = new Set<number>();
    for (let i = 0; i < firedCount; i += 1) {
      const idx = sortedIndices[i]!;
      const firedEntry = this.#nodesByIndex[idx]!;
      firedNodeIds[i] = firedEntry.lockfileNode.id;

      // Mirror cross-subsystem excitation onto any other affected FDM band
      if (firedEntry.band !== entry.band && !touchedOtherBands.has(firedEntry.band)) {
        touchedOtherBands.add(firedEntry.band);
        const snap = this.scheduler.inspectNode(idx);
        this.bus.emitWave(firedEntry.band, snap.energy, 0.0);
      }
    }

    const waveDurationMs = performance.now() - startMs;
    return Object.freeze({
      firedNodeIds: Object.freeze(firedNodeIds),
      cancelled: firedCount === 0,
      waveDurationMs,
      firedCount,
      totalNodes: this.totalNodes,
    });
  }

  /**
   * Walks ONLY `firedNodeIds` in strict bottom-up topological order (leaves first,
   * parents after). Checks `store.hasBlob(node.sha512, node.executable)` first
   * for O(1) CAS hits and invokes `compileFn` only on true CAS misses, storing
   * newly compiled artifacts into `CasV11Store`.
   */
  async executeWave(
    firedNodeIds: readonly string[],
    store: CasV11Store,
    compileFn: WaveCompileFn,
  ): Promise<WaveExecutionResult> {
    const startMs = performance.now();

    // Deduplicate and order strictly bottom-up via post-order nodeIndex
    const uniqueEntries: DagNodeEntry[] = [];
    const seen = new Set<number>();
    for (let i = 0; i < firedNodeIds.length; i += 1) {
      const id = firedNodeIds[i]!;
      const entry = this.#nodesById.get(id);
      if (entry === undefined) {
        throw new Error(`Unknown lockfile node ID in executeWave: "${id}"`);
      }
      if (!seen.has(entry.nodeIndex)) {
        seen.add(entry.nodeIndex);
        uniqueEntries.push(entry);
      }
    }
    uniqueEntries.sort((a, b) => a.nodeIndex - b.nodeIndex);

    const compiledIds: string[] = [];
    const casHitIds: string[] = [];

    for (let i = 0; i < uniqueEntries.length; i += 1) {
      const entry = uniqueEntries[i]!;
      const node = entry.lockfileNode;

      // O(1) v11 CAS hit check -> skip compile completely
      if (store.hasBlob(node.sha512, node.executable)) {
        casHitIds.push(node.id);
        continue;
      }

      // True CAS miss -> compile & persist into CasV11Store
      const artifact = await compileFn(node, entry);
      store.putBlob(artifact, {
        kind: toCasBlobKind(node.kind),
        executable: node.executable,
        meta: {
          id: node.id,
          version: node.version,
          target: node.target,
          lockfileSha512: node.sha512,
        },
      });
      compiledIds.push(node.id);
    }

    const totalMs = performance.now() - startMs;
    const processed = casHitIds.length + compiledIds.length;
    const casHitRate = processed === 0 ? 1.0 : casHitIds.length / processed;

    return Object.freeze({
      compiledIds: Object.freeze(compiledIds),
      casHitIds: Object.freeze(casHitIds),
      totalMs,
      casHitRate,
    });
  }
}
