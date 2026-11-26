/**
 * @file packages/governance/index.mts
 * @description Unified Governance entrypoint (Ed25519 Verified Registry + Model Armor + Gateway Policy).
 *
 * Strictly `erasableSyntaxOnly: true`, zero npm runtime dependencies.
 */

export {
  VerifiedArtifactRegistry,
  canonicalizeJson,
  defaultAwesomeOsRootKey,
  deriveEd25519KeyFromSeed,
  generateEd25519RootKey,
  sha512HexUtf8,
  signAgentCard,
  signManifestEd25519,
  verifyAgentCard,
  verifyManifestEd25519,
  type AgentCardPayload,
  type Ed25519KeyMaterial,
  type ManifestVerificationResult,
  type SignedAgentCard,
  type VerifiedBootSignature,
} from './verified-registry.mts';

export {
  AgentGatewayPolicy,
  BloomOodGate,
  ModelArmor,
  type AgentGatewayConfig,
  type ArmorInspectionResult,
  type BloomOodResult,
  type GatewayPolicyDecision,
  type GatewayRequestContext,
  type InjectionMatch,
  type InjectionSeverity,
  type PromptInjectionScanResult,
  type RedactionKind,
  type RedactionResult,
} from './model-armor.mts';
