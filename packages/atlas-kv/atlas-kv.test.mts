import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  AtlasKvSection,
  benchmarkAtlasKv,
  HiKvpArena,
  kg2kvEncodeInto,
  type KgTriple,
} from './index.mts';

describe('packages/atlas-kv — KG2KV + HiKVP Hierarchical Parametric Memory', () => {
  it('encodes KG triples deterministically and enforces CapabilityToken tenant isolation', () => {
    const triples: KgTriple[] = [
      { id: 1, head: 'Inkling-2-Mini', relation: 'uses_quantization', tail: 'NVFP4-MXFP8', tenantScope: 'public' },
      { id: 2, head: 'Xenon-1B', relation: 'uses_memory', tail: 'AtlasKV-HiKVP', tenantScope: 'public' },
      { id: 3, head: 'Project-Aurora', relation: 'secret_key', tail: 'top-secret-alpha', tenantScope: 'tenant-a' },
      { id: 4, head: 'Holo-v11', relation: 'hash_algo', tail: 'SHA-512', tenantScope: 'public' },
    ];

    const arena = HiKvpArena.fromTriples(triples, 64, 2);
    const q = new Float32Array(64);
    kg2kvEncodeInto('Project-Aurora secret_key top-secret-alpha', 64, q, 0);

    // Public reader cannot see tenant-a memory cluster
    const publicHits = arena.queryHiKvp(q, 2, 4, new Set(['public']));
    assert.ok(!publicHits.some((h) => h.id === 3));

    // Tenant-a reader retrieves private triple #3 as top hit
    const privateHits = arena.queryHiKvp(q, 2, 1, new Set(['public', 'tenant-a']));
    assert.equal(privateHits[0]?.id, 3);
    assert.ok((privateHits[0]?.score ?? 0) > 0.9);
  });

  it('beats flat O(N) scan with sub-linear HiKVP clustering and exports B8G Section schema', () => {
    const bench = benchmarkAtlasKv({
      tripleCount: 20_000,
      dim: 32,
      numClusters: 64,
      coarseTopC: 2,
      queries: 20,
    });
    assert.equal(bench.tripleCount, 20_000);
    assert.ok(bench.vramReductionPct > 80, `Expected >80% active working-set reduction, got ${bench.vramReductionPct}%`);
    assert.ok(bench.avgTop1Recall >= 0.85, `Expected high recall, got ${bench.avgTop1Recall}`);
    assert.equal(AtlasKvSection.id, 'awesome-os/atlas-kv');
    assert.equal(AtlasKvSection.substrate.kind, 'uvs-cas');
  });
});
