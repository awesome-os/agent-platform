/**
 * @file sections/atlas-kv/index.mts
 * @description Awesome-OS AtlasKV Alignment Section (`KG2KV` + `HiKVP` Hierarchical Key-Value Pruning
 * over flat `Float32Array` / `B8G_ARENA_V1` arenas with `CapabilityToken` tenant isolation).
 *
 * Inspired by AtlasKV (arXiv:2510.17934, ICLR):
 *   - `kg2kvEncode()` converts `(head, relation, tail)` KG triples into L2-normalized KV vectors.
 *   - `HiKvpArena` organizes keys into a 2-level coarse->leaf cluster hierarchy in contiguous
 *     `Float32Array` memory so inference touches only active top-C clusters (sub-linear time & VRAM).
 *   - `benchmarkAtlasKv()` measures O(N) Flat Scan vs. Sub-Linear HiKVP pruning speedup & working-set bytes.
 *
 * Strictly `erasableSyntaxOnly: true` and zero `node:*` imports.
 */

import {
  defineSection,
  f,
  p,
  s,
  type InferSection,
} from '../schema/index.mts'

export interface KgTriple {
  readonly id: number
  readonly head: string
  readonly relation: string
  readonly tail: string
  readonly tenantScope?: string
}

export interface HiKvpHit {
  readonly id: number
  readonly score: number
  readonly clusterId: number
  readonly tenantScope: string
}

export interface AtlasKvBenchResult {
  readonly tripleCount: number
  readonly dim: number
  readonly numClusters: number
  readonly queries: number
  readonly flatMs: number
  readonly hiKvpMs: number
  readonly speedupX: number
  readonly totalKvBytes: number
  readonly activeClusterBytes: number
  readonly vramReductionPct: number
  readonly avgTop1Recall: number
}

function hashFnv1a32(str: string, seed: number = 0x811c9dc5): number {
  let h = seed >>> 0
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i)
    h = Math.imul(h, 0x01000193) >>> 0
  }
  return h >>> 0
}

/**
 * KG2KV deterministic feature-hash encoder: maps `(head, relation, tail)` into an L2-normalized
 * vector written in-place into `out` at `offset` (zero per-triple allocations in inner loops).
 */
export function kg2kvEncodeInto(
  text: string,
  dim: number,
  out: Float32Array,
  offset: number
): void {
  for (let d = 0; d < dim; d += 1) out[offset + d] = 0
  const tokens = text.toLowerCase().split(/[^a-z0-9_]+/).filter((t) => t.length > 0)
  for (let i = 0; i < tokens.length; i += 1) {
    const tok = tokens[i]!
    const h1 = hashFnv1a32(tok, 0x811c9dc5)
    const h2 = hashFnv1a32(tok, 0x9e3779b9)
    const idx1 = h1 % dim
    const idx2 = h2 % dim
    out[offset + idx1] = out[offset + idx1]! + ((h1 & 1) === 0 ? 1 : -1)
    out[offset + idx2] = out[offset + idx2]! + ((h2 & 1) === 0 ? 0.5 : -0.5)
  }
  let normSq = 0
  for (let d = 0; d < dim; d += 1) {
    const v = out[offset + d]!
    normSq += v * v
  }
  const inv = normSq > 1e-12 ? 1 / Math.sqrt(normSq) : 0
  for (let d = 0; d < dim; d += 1) {
    out[offset + d] = out[offset + d]! * inv
  }
}

export function dotFlat(
  a: Float32Array,
  aOff: number,
  b: Float32Array,
  bOff: number,
  dim: number
): number {
  let acc = 0
  for (let d = 0; d < dim; d += 1) {
    acc += a[aOff + d]! * b[bOff + d]!
  }
  return acc
}

/**
 * Flat `Float32Array` 2-level Hierarchical Key-Value Pruner (`HiKVP`) aligned to `B8G_ARENA_V1`.
 * Keeps all cold KV keys in contiguous flat arrays and routes queries via coarse cluster centroids
 * so only active top-C cluster slices are scanned (sub-linear time & working-set VRAM).
 */
export class HiKvpArena {
  readonly dim: number
  readonly numClusters: number
  readonly count: number
  readonly keysF32: Float32Array
  readonly centroidsF32: Float32Array
  readonly clusterOffsets: Int32Array
  readonly clusterLengths: Int32Array
  readonly itemIds: Int32Array
  readonly itemScopes: string[]

  constructor(
    dim: number,
    numClusters: number,
    keysF32: Float32Array,
    centroidsF32: Float32Array,
    clusterOffsets: Int32Array,
    clusterLengths: Int32Array,
    itemIds: Int32Array,
    itemScopes: string[]
  ) {
    this.dim = dim
    this.numClusters = numClusters
    this.count = itemIds.length
    this.keysF32 = keysF32
    this.centroidsF32 = centroidsF32
    this.clusterOffsets = clusterOffsets
    this.clusterLengths = clusterLengths
    this.itemIds = itemIds
    this.itemScopes = itemScopes
  }

  /**
   * Builds a clustered `HiKvpArena` from `KgTriple[]` using deterministic coarse-centroid assignment.
   */
  static fromTriples(triples: readonly KgTriple[], dim: number = 32, numClusters: number = 16): HiKvpArena {
    const n = triples.length
    const cCount = Math.max(1, Math.min(numClusters, Math.max(1, n)))
    const rawKeys = new Float32Array(n * dim)
    for (let i = 0; i < n; i += 1) {
      const t = triples[i]!
      kg2kvEncodeInto(`${t.head} ${t.relation} ${t.tail}`, dim, rawKeys, i * dim)
    }

    // Initialize centroids from evenly spaced triples
    const centroids = new Float32Array(cCount * dim)
    for (let c = 0; c < cCount; c += 1) {
      const srcIdx = Math.floor((c * n) / cCount)
      for (let d = 0; d < dim; d += 1) {
        centroids[c * dim + d] = rawKeys[srcIdx * dim + d]!
      }
    }

    // Assign each triple to its nearest centroid
    const buckets: number[][] = Array.from({ length: cCount }, () => [])
    for (let i = 0; i < n; i += 1) {
      let bestC = 0
      let bestScore = -Infinity
      const off = i * dim
      for (let c = 0; c < cCount; c += 1) {
        const s = dotFlat(rawKeys, off, centroids, c * dim, dim)
        if (s > bestScore) {
          bestScore = s
          bestC = c
        }
      }
      buckets[bestC]!.push(i)
    }

    // Recompute L2-normalized centroids and pack contiguous cluster-ordered keysF32
    const packedKeys = new Float32Array(n * dim)
    const offsets = new Int32Array(cCount)
    const lengths = new Int32Array(cCount)
    const ids = new Int32Array(n)
    const scopes: string[] = new Array<string>(n)

    let cursor = 0
    for (let c = 0; c < cCount; c += 1) {
      const b = buckets[c]!
      offsets[c] = cursor
      lengths[c] = b.length
      const cOff = c * dim
      for (let d = 0; d < dim; d += 1) centroids[cOff + d] = 0

      for (const origIdx of b) {
        const srcOff = origIdx * dim
        const dstOff = cursor * dim
        for (let d = 0; d < dim; d += 1) {
          const v = rawKeys[srcOff + d]!
          packedKeys[dstOff + d] = v
          centroids[cOff + d] = centroids[cOff + d]! + v
        }
        ids[cursor] = triples[origIdx]!.id
        scopes[cursor] = triples[origIdx]!.tenantScope ?? 'public'
        cursor += 1
      }

      let normSq = 0
      for (let d = 0; d < dim; d += 1) {
        const v = centroids[cOff + d]!
        normSq += v * v
      }
      const inv = normSq > 1e-12 ? 1 / Math.sqrt(normSq) : 0
      for (let d = 0; d < dim; d += 1) centroids[cOff + d] = centroids[cOff + d]! * inv
    }

    return new HiKvpArena(dim, cCount, packedKeys, centroids, offsets, lengths, ids, scopes)
  }

  /**
   * Builds a synthetic clustered `HiKvpArena` directly in `Float32Array` for large-scale benchmarks (e.g. 100k+ triples).
   */
  static buildSynthetic(tripleCount: number, dim: number = 32, numClusters: number = 128): HiKvpArena {
    const cCount = Math.max(1, Math.min(numClusters, tripleCount))
    const centroids = new Float32Array(cCount * dim)
    const packedKeys = new Float32Array(tripleCount * dim)
    const offsets = new Int32Array(cCount)
    const lengths = new Int32Array(cCount)
    const ids = new Int32Array(tripleCount)
    const scopes: string[] = new Array<string>(tripleCount)

    let seed = 0x1337beef
    const nextRand = (): number => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
      return (seed / 0xffffffff) * 2 - 1
    }

    for (let c = 0; c < cCount; c += 1) {
      const cOff = c * dim
      let normSq = 0
      for (let d = 0; d < dim; d += 1) {
        const v = nextRand()
        centroids[cOff + d] = v
        normSq += v * v
      }
      const inv = 1 / Math.sqrt(normSq || 1)
      for (let d = 0; d < dim; d += 1) centroids[cOff + d] = centroids[cOff + d]! * inv
    }

    const perCluster = Math.floor(tripleCount / cCount)
    let cursor = 0
    for (let c = 0; c < cCount; c += 1) {
      const len = c === cCount - 1 ? tripleCount - cursor : perCluster
      offsets[c] = cursor
      lengths[c] = len
      const cOff = c * dim
      for (let j = 0; j < len; j += 1) {
        const dstOff = cursor * dim
        let normSq = 0
        for (let d = 0; d < dim; d += 1) {
          const v = centroids[cOff + d]! + nextRand() * 0.12
          packedKeys[dstOff + d] = v
          normSq += v * v
        }
        const inv = 1 / Math.sqrt(normSq || 1)
        for (let d = 0; d < dim; d += 1) packedKeys[dstOff + d] = packedKeys[dstOff + d]! * inv
        ids[cursor] = cursor
        scopes[cursor] = c % 4 === 0 ? 'tenant:private' : 'public'
        cursor += 1
      }
    }

    return new HiKvpArena(dim, cCount, packedKeys, centroids, offsets, lengths, ids, scopes)
  }

  /**
   * Baseline O(N) flat scan across all triples.
   */
  queryFlatScan(
    queryVec: Float32Array,
    topK: number = 1,
    allowedScopes?: ReadonlySet<string>
  ): HiKvpHit[] {
    const best: HiKvpHit[] = []
    for (let i = 0; i < this.count; i += 1) {
      const scope = this.itemScopes[i]!
      if (allowedScopes && !allowedScopes.has(scope)) continue
      const score = dotFlat(this.keysF32, i * this.dim, queryVec, 0, this.dim)
      if (best.length < topK || score > best[best.length - 1]!.score) {
        best.push({ id: this.itemIds[i]!, score, clusterId: -1, tenantScope: scope })
        best.sort((a, b) => b.score - a.score)
        if (best.length > topK) best.length = topK
      }
    }
    return best
  }

  /**
   * Sub-linear HiKVP hierarchical pruned query:
   *   Layer 1 (coarse): score `numClusters` centroids, keep top-`coarseTopC`.
   *   Layer 2 (fine):   scan ONLY leaf keys inside those top-`coarseTopC` clusters (+ tenant capability check).
   */
  queryHiKvp(
    queryVec: Float32Array,
    coarseTopC: number = 2,
    leafTopK: number = 1,
    allowedScopes?: ReadonlySet<string>
  ): HiKvpHit[] {
    const topClusters: Array<{ c: number; score: number }> = []
    const kC = Math.max(1, Math.min(coarseTopC, this.numClusters))
    for (let c = 0; c < this.numClusters; c += 1) {
      const score = dotFlat(this.centroidsF32, c * this.dim, queryVec, 0, this.dim)
      if (topClusters.length < kC || score > topClusters[topClusters.length - 1]!.score) {
        topClusters.push({ c, score })
        topClusters.sort((a, b) => b.score - a.score)
        if (topClusters.length > kC) topClusters.length = kC
      }
    }

    const best: HiKvpHit[] = []
    for (const tc of topClusters) {
      const start = this.clusterOffsets[tc.c]!
      const end = start + this.clusterLengths[tc.c]!
      for (let i = start; i < end; i += 1) {
        const scope = this.itemScopes[i]!
        if (allowedScopes && !allowedScopes.has(scope)) continue
        const score = dotFlat(this.keysF32, i * this.dim, queryVec, 0, this.dim)
        if (best.length < leafTopK || score > best[best.length - 1]!.score) {
          best.push({ id: this.itemIds[i]!, score, clusterId: tc.c, tenantScope: scope })
          best.sort((a, b) => b.score - a.score)
          if (best.length > leafTopK) best.length = leafTopK
        }
      }
    }
    return best
  }
}

/**
 * Reproducible AtlasKV vs. Flat Scan benchmark (sub-linear speedup + active VRAM reduction + top-1 recall).
 */
export function benchmarkAtlasKv(opts: {
  readonly tripleCount?: number
  readonly dim?: number
  readonly numClusters?: number
  readonly coarseTopC?: number
  readonly queries?: number
} = {}): AtlasKvBenchResult {
  const tripleCount = opts.tripleCount ?? 50_000
  const dim = opts.dim ?? 32
  const numClusters = opts.numClusters ?? 128
  const coarseTopC = opts.coarseTopC ?? 2
  const queries = opts.queries ?? 40

  const arena = HiKvpArena.buildSynthetic(tripleCount, dim, numClusters)
  const queryVecs: Float32Array[] = []
  const expectedIds: number[] = []

  for (let q = 0; q < queries; q += 1) {
    const targetIdx = Math.floor(((q + 0.5) * tripleCount) / queries)
    const vec = new Float32Array(dim)
    for (let d = 0; d < dim; d += 1) {
      vec[d] = arena.keysF32[targetIdx * dim + d]!
    }
    queryVecs.push(vec)
    expectedIds.push(arena.itemIds[targetIdx]!)
  }

  const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now())

  const tFlat0 = now()
  for (let q = 0; q < queries; q += 1) {
    arena.queryFlatScan(queryVecs[q]!, 1)
  }
  const flatMs = Math.max(0.01, now() - tFlat0)

  const tHi0 = now()
  let recallHits = 0
  for (let q = 0; q < queries; q += 1) {
    const res = arena.queryHiKvp(queryVecs[q]!, coarseTopC, 1)
    if (res[0] && res[0].id === expectedIds[q]) recallHits += 1
  }
  const hiKvpMs = Math.max(0.01, now() - tHi0)

  const totalKvBytes = arena.keysF32.byteLength
  const activeClusterBytes =
    arena.centroidsF32.byteLength +
    Math.ceil((tripleCount / numClusters) * coarseTopC) * dim * 4
  const vramReductionPct = Math.round((1 - activeClusterBytes / totalKvBytes) * 1000) / 10

  return {
    tripleCount,
    dim,
    numClusters,
    queries,
    flatMs: Math.round(flatMs * 100) / 100,
    hiKvpMs: Math.round(hiKvpMs * 100) / 100,
    speedupX: Math.round((flatMs / hiKvpMs) * 10) / 10,
    totalKvBytes,
    activeClusterBytes,
    vramReductionPct,
    avgTop1Recall: Math.round((recallHits / queries) * 1000) / 1000,
  }
}

export interface RecordAtlasKvBenchInput {
  readonly tripleCount: number
  readonly clusterCount: number
  readonly activeVramBytes: number
  readonly speedupVsFlatX: number
  readonly recall: number
}

export const AtlasKvSection = defineSection({
  id: 'awesome-os/atlas-kv',
  version: '1.0.0',
  substrate: s.uvsCas({ prefix: 'atlas-kv' }),

  fields: {
    status: f.string({ maxBytes: 32 }).default('ready'),
    tripleCount: f.int32().default(0),
    clusterCount: f.int32().default(0),
    activeVramBytes: f.int32().default(0),
    speedupVsFlatX: f.float64().default(1),
    recall: f.float64().default(1),
    epoch: f.int32().default(1),
  },

  protocols: {
    events: p.output('b8g-mesh', 'atlas-kv/events'),
  },

  hydrate(ctx) {
    const cur = ctx.readState()
    return {
      status: cur.status || 'ready',
      epoch: Math.max(1, cur.epoch),
    }
  },

  ops: {
    recordBenchmark(ctx, input: RecordAtlasKvBenchInput) {
      const cur = ctx.readState()
      return ctx.patchState({
        status: 'benchmarked',
        tripleCount: Math.max(0, Math.trunc(input.tripleCount)),
        clusterCount: Math.max(0, Math.trunc(input.clusterCount)),
        activeVramBytes: Math.max(0, Math.trunc(input.activeVramBytes)),
        speedupVsFlatX: input.speedupVsFlatX,
        recall: input.recall,
        epoch: cur.epoch + 1,
      })
    },
  },
})

export type AtlasKvState = InferSection<typeof AtlasKvSection>
