/**
 * @file intersections/b8g-moe-gate/index.mts
 * @description The First Real Boundary-Verified MoE Router & Gate Intersection.
 *
 * Every MoE expert is an isolated `B8G-SECT-1` Section guarded by hard ingress and
 * egress mathematical, capability, and symbolic boundary gates:
 *   1. Ingress Gate: finite-value check, L2 energy bound, `CapabilityToken` scope check,
 *      deterministic top-K & loop-pass cap.
 *   2. Execution: Sparse Top-K MoE + Multi-Pass Looped Core with Rank-r Gated Inter-Pass
 *      blending and Block Attention Residual depth-wise fusion.
 *   3. Egress Gate: finite-value check, L2 energy/Lipschitz bound, optional symbolic safe gate,
 *      and automatic quarantine of any violating expert (zero residual-stream poisoning).
 *
 * Strictly `erasableSyntaxOnly: true` & zero-dependency compliant.
 */

import { defineSection, f, s, type ExecutionContext, type StreamFrame } from '../schema/index.mts'

export const B8G_MOE_GATE_CONTEXT: ExecutionContext = {
  boundary: 'in-process',
  display: 'headless',
  lifetime: 'oneshot',
  transport: 'arena-sab',
  capability: {
    tenantId: 'local-user',
    scopes: ['moe.route'],
    epoch: 1,
    privateData: true,
  },
  locality: 'local-arena',
}

export const B8gMoeGateSection = defineSection({
  id: 'b8g/moe-gate',
  version: '1.0.0',
  substrate: s.sabRing({ ringBytes: 65536 }),
  context: B8G_MOE_GATE_CONTEXT,
  fields: {
    hiddenSize: f.int32(),
    topK: f.int32(),
    loopPasses: f.int32(),
    totalDispatches: f.int32(),
    totalQuarantines: f.int32(),
    totalCapDenials: f.int32(),
  },
})

export interface CapabilityToken {
  readonly tenantId: string
  readonly allowedScopes: readonly string[]
}

export type GateStatus =
  | 'OK'
  | 'CAPABILITY_DENIED'
  | 'NON_FINITE_INPUT'
  | 'INGRESS_NORM_EXCEEDED'
  | 'NON_FINITE_OUTPUT'
  | 'EGRESS_NORM_EXCEEDED'
  | 'SYMBOLIC_GATE_REJECTED'

export interface SymbolicGateResult {
  readonly ok: boolean
  readonly reason?: string
}

export interface VerifiedMoeExpert {
  readonly id: number
  readonly name: string
  readonly requiredScope: string
  /** Router prototype vector (same dim as hiddenSize) for deterministic dot-product gating. */
  readonly routerKey: Float32Array
  /** Pure section forward pass: writes expert output into `out` (zero heap alloc). */
  readonly forward: (input: Float32Array, out: Float32Array, passIndex: number) => void
  /** Optional symbolic safe-gate verifier (e.g., V8 syntax / R <= M / C11 ABI / schema check). */
  readonly symbolicGate?: (output: Float32Array, passIndex: number) => SymbolicGateResult
}

export interface VerifiedMoeConfig {
  readonly hiddenSize: number
  readonly topK: number
  readonly loopPasses: number
  readonly maxIngressL2Norm: number
  readonly maxEgressL2Norm: number
  readonly loopBlendAlpha: number
}

export interface ExpertGateAudit {
  readonly expertId: number
  readonly expertName: string
  readonly passIndex: number
  readonly routerWeight: number
  readonly ingressNorm: number
  readonly egressNorm: number
  readonly status: GateStatus
  readonly reason?: string
}

export interface VerifiedMoeDispatchResult {
  readonly ok: boolean
  readonly output: Float32Array
  readonly passesExecuted: number
  readonly activeExpertsExecuted: number
  readonly quarantinedCount: number
  readonly capabilityDenials: number
  readonly audits: readonly ExpertGateAudit[]
}

function l2Norm(vec: Float32Array): { readonly finite: boolean; readonly norm: number } {
  let sumSq = 0
  for (let i = 0; i < vec.length; i += 1) {
    const v = vec[i]!
    if (!Number.isFinite(v)) return { finite: false, norm: Number.NaN }
    sumSq += v * v
  }
  return { finite: true, norm: Math.sqrt(sumSq) }
}

function dot(a: Float32Array, b: Float32Array): number {
  const n = Math.min(a.length, b.length)
  let s = 0
  for (let i = 0; i < n; i += 1) s += a[i]! * b[i]!
  return s
}

export class B8gVerifiedMoERouter {
  readonly config: VerifiedMoeConfig
  readonly #experts: VerifiedMoeExpert[] = []
  readonly #quarantinedIds = new Set<number>()
  #totalDispatches = 0
  #totalQuarantines = 0
  #totalCapDenials = 0

  constructor(config: VerifiedMoeConfig) {
    this.config = config
  }

  registerExpert(expert: VerifiedMoeExpert): void {
    this.#experts.push(expert)
  }

  isQuarantined(expertId: number): boolean {
    return this.#quarantinedIds.has(expertId)
  }

  resetQuarantine(): void {
    this.#quarantinedIds.clear()
  }

  getAuditTotals(): {
    readonly totalDispatches: number
    readonly totalQuarantines: number
    readonly totalCapDenials: number
    readonly quarantinedExpertIds: readonly number[]
  } {
    return {
      totalDispatches: this.#totalDispatches,
      totalQuarantines: this.#totalQuarantines,
      totalCapDenials: this.#totalCapDenials,
      quarantinedExpertIds: Array.from(this.#quarantinedIds),
    }
  }

  /**
   * Executes a boundary-verified sparse MoE + multi-pass looped forward step with
   * Block Attention Residual fusion across loop passes.
   */
  dispatch(input: Float32Array, token: CapabilityToken): VerifiedMoeDispatchResult {
    this.#totalDispatches += 1
    const h = this.config.hiddenSize
    const audits: ExpertGateAudit[] = []

    // 1. Global Ingress Boundary Gate (finite + energy bound)
    const inCheck = l2Norm(input)
    if (!inCheck.finite) {
      audits.push({
        expertId: -1,
        expertName: 'INGRESS_GATE',
        passIndex: 0,
        routerWeight: 0,
        ingressNorm: Number.NaN,
        egressNorm: 0,
        status: 'NON_FINITE_INPUT',
        reason: 'Ingress tensor contains NaN or Infinity',
      })
      return {
        ok: false,
        output: new Float32Array(h),
        passesExecuted: 0,
        activeExpertsExecuted: 0,
        quarantinedCount: 0,
        capabilityDenials: 0,
        audits,
      }
    }

    if (inCheck.norm > this.config.maxIngressL2Norm) {
      audits.push({
        expertId: -1,
        expertName: 'INGRESS_GATE',
        passIndex: 0,
        routerWeight: 0,
        ingressNorm: inCheck.norm,
        egressNorm: 0,
        status: 'INGRESS_NORM_EXCEEDED',
        reason: `Ingress L2 norm ${inCheck.norm.toFixed(2)} > max ${this.config.maxIngressL2Norm}`,
      })
      return {
        ok: false,
        output: new Float32Array(h),
        passesExecuted: 0,
        activeExpertsExecuted: 0,
        quarantinedCount: 0,
        capabilityDenials: 0,
        audits,
      }
    }

    // Working buffers (zero per-expert heap allocation inside loop)
    const current = new Float32Array(h)
    current.set(input.subarray(0, h))
    const passAccum = new Float32Array(h)
    const expertScratch = new Float32Array(h)
    const blockHistory: Float32Array[] = [new Float32Array(current)]

    let activeExpertsExecuted = 0
    let quarantinedCount = 0
    let capabilityDenials = 0
    const passes = Math.max(1, this.config.loopPasses)

    for (let pass = 0; pass < passes; pass += 1) {
      const curCheck = l2Norm(current)
      // Score all non-quarantined experts deterministically
      const scored: Array<{ readonly expert: VerifiedMoeExpert; readonly score: number }> = []
      for (const exp of this.#experts) {
        if (this.#quarantinedIds.has(exp.id)) continue
        scored.push({ expert: exp, score: dot(current, exp.routerKey) })
      }
      scored.sort((a, b) => b.score - a.score)

      // Select topK eligible experts with CapabilityToken boundary check
      const selected: Array<{ readonly expert: VerifiedMoeExpert; readonly score: number }> = []
      for (const item of scored) {
        if (selected.length >= this.config.topK) break
        const allowed =
          token.allowedScopes.includes('*') ||
          token.allowedScopes.includes(item.expert.requiredScope)
        if (!allowed) {
          capabilityDenials += 1
          this.#totalCapDenials += 1
          audits.push({
            expertId: item.expert.id,
            expertName: item.expert.name,
            passIndex: pass,
            routerWeight: 0,
            ingressNorm: curCheck.norm,
            egressNorm: 0,
            status: 'CAPABILITY_DENIED',
            reason: `Tenant '${token.tenantId}' lacks scope '${item.expert.requiredScope}'`,
          })
          continue
        }
        selected.push(item)
      }

      // Softmax weights over selected experts
      let maxScore = -Infinity
      for (const s of selected) if (s.score > maxScore) maxScore = s.score
      const exps = selected.map((s) => Math.exp(s.score - (Number.isFinite(maxScore) ? maxScore : 0)))
      const sumExp = exps.reduce((a, b) => a + b, 0) || 1

      passAccum.fill(0)
      let acceptedWeightSum = 0

      for (let i = 0; i < selected.length; i += 1) {
        const exp = selected[i]!.expert
        const w = exps[i]! / sumExp
        expertScratch.fill(0)
        exp.forward(current, expertScratch, pass)

        // 2. Egress Boundary Gate (finite + energy bound + symbolic safe gate)
        const outCheck = l2Norm(expertScratch)
        if (!outCheck.finite) {
          this.#quarantinedIds.add(exp.id)
          quarantinedCount += 1
          this.#totalQuarantines += 1
          audits.push({
            expertId: exp.id,
            expertName: exp.name,
            passIndex: pass,
            routerWeight: w,
            ingressNorm: curCheck.norm,
            egressNorm: Number.NaN,
            status: 'NON_FINITE_OUTPUT',
            reason: 'Expert produced NaN/Inf; quarantined to protect residual stream',
          })
          continue
        }

        if (outCheck.norm > this.config.maxEgressL2Norm) {
          this.#quarantinedIds.add(exp.id)
          quarantinedCount += 1
          this.#totalQuarantines += 1
          audits.push({
            expertId: exp.id,
            expertName: exp.name,
            passIndex: pass,
            routerWeight: w,
            ingressNorm: curCheck.norm,
            egressNorm: outCheck.norm,
            status: 'EGRESS_NORM_EXCEEDED',
            reason: `Expert L2 norm ${outCheck.norm.toFixed(2)} > max ${this.config.maxEgressL2Norm}; quarantined`,
          })
          continue
        }

        if (exp.symbolicGate) {
          const sym = exp.symbolicGate(expertScratch, pass)
          if (!sym.ok) {
            this.#quarantinedIds.add(exp.id)
            quarantinedCount += 1
            this.#totalQuarantines += 1
            audits.push({
              expertId: exp.id,
              expertName: exp.name,
              passIndex: pass,
              routerWeight: w,
              ingressNorm: curCheck.norm,
              egressNorm: outCheck.norm,
              status: 'SYMBOLIC_GATE_REJECTED',
              reason: sym.reason ?? 'Symbolic safe gate rejected expert output',
            })
            continue
          }
        }

        // Accepted! Accumulate weighted expert contribution
        acceptedWeightSum += w
        activeExpertsExecuted += 1
        for (let j = 0; j < h; j += 1) {
          passAccum[j] = passAccum[j]! + w * expertScratch[j]!
        }
        audits.push({
          expertId: exp.id,
          expertName: exp.name,
          passIndex: pass,
          routerWeight: Number(w.toFixed(4)),
          ingressNorm: Number(curCheck.norm.toFixed(4)),
          egressNorm: Number(outCheck.norm.toFixed(4)),
          status: 'OK',
        })
      }

      // Normalize if some experts were quarantined
      if (acceptedWeightSum > 0 && acceptedWeightSum < 0.9999) {
        const inv = 1 / acceptedWeightSum
        for (let j = 0; j < h; j += 1) passAccum[j] = passAccum[j]! * inv
      }

      // 3. Block Attention Residual + Gated Inter-Pass Loop Blending
      const alpha = this.config.loopBlendAlpha
      const histWeights = blockHistory.map((blk) => Math.exp(dot(current, blk) / Math.sqrt(h)))
      const histSum = histWeights.reduce((a, b) => a + b, 0) || 1
      for (let j = 0; j < h; j += 1) {
        let attnRes = 0
        for (let b = 0; b < blockHistory.length; b += 1) {
          attnRes += (histWeights[b]! / histSum) * blockHistory[b]![j]!
        }
        current[j] = (1 - alpha) * attnRes + alpha * passAccum[j]!
      }
      blockHistory.push(new Float32Array(current))
    }

    return {
      ok: true,
      output: current,
      passesExecuted: passes,
      activeExpertsExecuted,
      quarantinedCount,
      capabilityDenials,
      audits,
    }
  }
}

export interface MoeGateTransformOptions {
  readonly router?: B8gVerifiedMoERouter
  readonly config?: Partial<VerifiedMoeConfig>
  readonly experts?: readonly VerifiedMoeExpert[]
  readonly token?: CapabilityToken
  readonly source?: string
}

const DEFAULT_MOE_GATE_CONFIG: VerifiedMoeConfig = {
  hiddenSize: 4,
  topK: 2,
  loopPasses: 1,
  maxIngressL2Norm: 1e4,
  maxEgressL2Norm: 1e4,
  loopBlendAlpha: 0.5,
}

const DEFAULT_MOE_GATE_TOKEN: CapabilityToken = {
  tenantId: 'local-user',
  allowedScopes: ['*'],
}

/**
 * Creates a pure Web `TransformStream` that runs boundary-verified MoE routing
 * at whole-tensor `Float32Array` chunk granularity per incoming `StreamFrame`.
 */
export function createMoeGateTransform(
  gateOpts?: MoeGateTransformOptions
): TransformStream<StreamFrame<Float32Array>, StreamFrame<VerifiedMoeDispatchResult>> {
  const router =
    gateOpts?.router ??
    new B8gVerifiedMoERouter({
      ...DEFAULT_MOE_GATE_CONFIG,
      ...gateOpts?.config,
    })

  if (!gateOpts?.router && gateOpts?.experts) {
    for (const expert of gateOpts.experts) {
      router.registerExpert(expert)
    }
  }

  const token = gateOpts?.token ?? DEFAULT_MOE_GATE_TOKEN
  const source = gateOpts?.source ?? 'b8g/moe-gate'

  return new TransformStream<StreamFrame<Float32Array>, StreamFrame<VerifiedMoeDispatchResult>>({
    transform(frame, controller) {
      const result = router.dispatch(frame.payload, token)
      controller.enqueue({
        kind: result.ok ? 'data' : 'error',
        source,
        minPersona: result.ok ? frame.minPersona : 'devops',
        payload: result,
        ts: Date.now(),
        ...(frame.violations ? { violations: frame.violations } : {}),
      })
    },
  })
}
