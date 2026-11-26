/**
 * @file tutorials/03-atlaskv-hikvp-memory.mts
 * @description Tutorial 03: AtlasKV `KG2KV` encoding + `HiKvpArena` 2-level hierarchical
 * key-value pruning with `CapabilityToken` tenant scope isolation and benchmark speedup.
 */

import assert from 'node:assert/strict';
import {
  benchmarkAtlasKv,
  HiKvpArena,
  kg2kvEncodeInto,
  type KgTriple,
} from '../packages/atlas-kv/index.mts';

function main(): void {
  console.log('=== Tutorial 03: AtlasKV KG2KV + HiKVP Hierarchical Memory ===');

  const triples: KgTriple[] = [
    { id: 1, head: 'sectium', relation: 'uses_abi', tail: 'B8G_ABI_MAGIC_V3', tenantScope: 'public' },
    { id: 2, head: 'b8g_xenon_1b', relation: 'externalizes_memory_via', tail: 'AtlasKV_HiKVP', tenantScope: 'public' },
    { id: 3, head: 'holo_compiler', relation: 'packs_objects_into', tail: 'b8p_cas_v11', tenantScope: 'public' },
    { id: 4, head: 'tenant_vault', relation: 'stores_secret_key', tail: 'ed25519_root_seed', tenantScope: 'tenant:private' },
    { id: 5, head: 'moe_gate', relation: 'enforces', tail: 'CapabilityToken_bounds', tenantScope: 'public' },
  ];

  const dim = 32;
  const arena = HiKvpArena.fromTriples(triples, dim, 3);
  const queryVec = new Float32Array(dim);
  kg2kvEncodeInto('tenant_vault stores_secret_key ed25519_root_seed', dim, queryVec, 0);

  // 1. Public tenant CANNOT retrieve 'tenant:private' triples
  const publicOnly = new Set<string>(['public']);
  const publicHits = arena.queryHiKvp(queryVec, 3, 2, publicOnly);
  assert.equal(publicHits.some((h) => h.id === 4), false, 'Private triple must be isolated from public scope');

  // 2. Authorized tenant retrieves exact private triple #4
  const authorized = new Set<string>(['public', 'tenant:private']);
  const authHits = arena.queryHiKvp(queryVec, 3, 2, authorized);
  assert.equal(authHits[0]?.id, 4, 'Authorized tenant must retrieve triple #4 as top-1 hit');
  console.log(`Capability isolation verified: public top1=${publicHits[0]?.id}, authorized top1=${authHits[0]?.id}`);

  // 3. Run reproducible sub-linear HiKVP vs O(N) flat scan benchmark
  const bench = benchmarkAtlasKv({
    tripleCount: 15_000,
    dim: 32,
    numClusters: 64,
    coarseTopC: 2,
    queries: 20,
  });

  console.log('\nAtlasKV Benchmark (15,000 triples, 64 clusters, top-2 pruning):');
  console.log(`  Flat Scan : ${bench.flatMs} ms (${(bench.totalKvBytes / 1024).toFixed(1)} KB)`);
  console.log(`  HiKVP     : ${bench.hiKvpMs} ms (${(bench.activeClusterBytes / 1024).toFixed(1)} KB active)`);
  console.log(`  Speedup   : ${bench.speedupX}x | VRAM reduction: ${bench.vramReductionPct}% | Top-1 Recall: ${bench.avgTop1Recall}`);
  assert.ok(bench.vramReductionPct > 90, 'Expected >90% active working-set VRAM reduction');
  assert.ok(bench.avgTop1Recall >= 0.95, 'Expected >=95% top-1 recall');
}

main();
