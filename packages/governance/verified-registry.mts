/**
 * @file packages/governance/verified-registry.mts
 * @description Awesome-OS Verified Boot & A2A Agent Card Ed25519 + SHA-512 Attestation
 * (adapted from `awesome-os/security/verified-boot.mts`).
 *
 * Strictly `erasableSyntaxOnly: true`, Node 22+ built-ins only.
 */

import {
  createHash,
  createPrivateKey,
  createPublicKey,
  generateKeyPairSync,
  sign,
  verify,
} from 'node:crypto';

export interface Ed25519KeyMaterial {
  readonly keyId: string;
  readonly publicKeyHex: string;
  readonly publicKeySpkiDerHex: string;
  readonly privateKeyPkcs8DerHex: string;
}

export interface VerifiedBootSignature {
  readonly scheme: 'ed25519-sha512-v1';
  readonly keyId: string;
  readonly publicKeyHex: string;
  readonly payloadSha512: string;
  readonly signatureHex: string;
  readonly signedAt: string;
}

export interface ManifestVerificationResult {
  readonly valid: boolean;
  readonly reason?: string;
  readonly payloadSha512: string;
}

export interface AgentCardPayload {
  readonly id: string;
  readonly name: string;
  readonly version: string;
  readonly url: string;
  readonly capabilities: readonly string[];
  readonly publicKeyHex?: string;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

export type SignedAgentCard = AgentCardPayload & {
  readonly signature: VerifiedBootSignature;
};

const ED25519_SPKI_PREFIX = Buffer.from('302a300506032b6570032100', 'hex');
const DEV_ROOT_SEED_LABEL = 'awesomeos-verified-boot-dev-root-v11-0x33564742';

/**
 * Canonicalizes any JSON value with lexicographically sorted object keys,
 * excluding `signature` and `undefined` properties.
 */
export function canonicalizeJson(value: unknown): string {
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((v) => canonicalizeJson(v)).join(',')}]`;
  }
  const obj = value as Record<string, unknown>;
  const keys = Object.keys(obj)
    .filter((k) => k !== 'signature' && obj[k] !== undefined)
    .sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalizeJson(obj[k])}`).join(',')}}`;
}

export function sha512HexUtf8(text: string): string {
  return createHash('sha512').update(text, 'utf8').digest('hex');
}

export function generateEd25519RootKey(keyId = 'awesomeos-root-2026'): Ed25519KeyMaterial {
  const { publicKey, privateKey } = generateKeyPairSync('ed25519');
  const spki = publicKey.export({ format: 'der', type: 'spki' });
  const pkcs8 = privateKey.export({ format: 'der', type: 'pkcs8' });
  const rawPub = spki.subarray(spki.length - 32);
  return {
    keyId,
    publicKeyHex: Buffer.from(rawPub).toString('hex'),
    publicKeySpkiDerHex: Buffer.from(spki).toString('hex'),
    privateKeyPkcs8DerHex: Buffer.from(pkcs8).toString('hex'),
  };
}

export function deriveEd25519KeyFromSeed(
  seedHex32: string,
  keyId = 'awesomeos-root-v11',
): Ed25519KeyMaterial {
  const seed = Buffer.from(seedHex32, 'hex');
  if (seed.length !== 32) {
    throw new Error('verified-registry: Ed25519 seed must be 32 bytes (64 hex chars)');
  }
  const pkcs8Prefix = Buffer.from('302e020100300506032b657004220420', 'hex');
  const pkcs8 = Buffer.concat([pkcs8Prefix, seed]);
  const priv = createPrivateKey({ key: pkcs8, format: 'der', type: 'pkcs8' });
  const pub = createPublicKey(priv);
  const spki = pub.export({ format: 'der', type: 'spki' });
  const rawPub = spki.subarray(spki.length - 32);
  return {
    keyId,
    publicKeyHex: Buffer.from(rawPub).toString('hex'),
    publicKeySpkiDerHex: Buffer.from(spki).toString('hex'),
    privateKeyPkcs8DerHex: pkcs8.toString('hex'),
  };
}

export function defaultAwesomeOsRootKey(): Ed25519KeyMaterial {
  const envSeed = process.env.AWESOMEOS_ED25519_SEED_HEX;
  const seedHex =
    envSeed && /^[0-9a-fA-F]{64}$/.test(envSeed)
      ? envSeed
      : createHash('sha256').update(DEV_ROOT_SEED_LABEL, 'utf8').digest('hex');
  return deriveEd25519KeyFromSeed(seedHex, 'awesomeos-root-v11');
}

export function signManifestEd25519<T extends Record<string, unknown>>(
  manifest: T,
  key: Ed25519KeyMaterial = defaultAwesomeOsRootKey(),
  signedAt: string = new Date().toISOString(),
): T & { readonly signature: VerifiedBootSignature } {
  const canonical = canonicalizeJson(manifest);
  const payloadSha512 = sha512HexUtf8(canonical);
  const priv = createPrivateKey({
    key: Buffer.from(key.privateKeyPkcs8DerHex, 'hex'),
    format: 'der',
    type: 'pkcs8',
  });
  const sigBuf = sign(null, Buffer.from(canonical, 'utf8'), priv);
  const signature: VerifiedBootSignature = Object.freeze({
    scheme: 'ed25519-sha512-v1',
    keyId: key.keyId,
    publicKeyHex: key.publicKeyHex,
    payloadSha512,
    signatureHex: sigBuf.toString('hex'),
    signedAt,
  });
  return { ...manifest, signature };
}

export function verifyManifestEd25519(
  signedManifest: Record<string, unknown>,
  trustedPublicKeyHex?: string,
): ManifestVerificationResult {
  const sigObj = signedManifest.signature as VerifiedBootSignature | undefined;
  if (!sigObj || sigObj.scheme !== 'ed25519-sha512-v1') {
    return { valid: false, reason: 'missing or unsupported signature scheme', payloadSha512: '' };
  }
  const pubHex = (trustedPublicKeyHex || sigObj.publicKeyHex || '').toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(pubHex)) {
    return { valid: false, reason: 'invalid 32-byte Ed25519 public key hex', payloadSha512: '' };
  }
  if (trustedPublicKeyHex && sigObj.publicKeyHex.toLowerCase() !== pubHex) {
    return { valid: false, reason: 'public key pin mismatch', payloadSha512: '' };
  }

  const canonical = canonicalizeJson(signedManifest);
  const payloadSha512 = sha512HexUtf8(canonical);
  if (payloadSha512 !== sigObj.payloadSha512) {
    return { valid: false, reason: 'canonical SHA-512 digest mismatch', payloadSha512 };
  }

  const spki = Buffer.concat([ED25519_SPKI_PREFIX, Buffer.from(pubHex, 'hex')]);
  const pub = createPublicKey({ key: spki, format: 'der', type: 'spki' });
  const ok = verify(
    null,
    Buffer.from(canonical, 'utf8'),
    pub,
    Buffer.from(sigObj.signatureHex, 'hex'),
  );
  return ok
    ? { valid: true, payloadSha512 }
    : { valid: false, reason: 'Ed25519 cryptographic signature verification failed', payloadSha512 };
}

export function signAgentCard(
  card: AgentCardPayload,
  key: Ed25519KeyMaterial = defaultAwesomeOsRootKey(),
  signedAt: string = new Date().toISOString(),
): SignedAgentCard {
  const withPub: AgentCardPayload = {
    ...card,
    publicKeyHex: card.publicKeyHex ?? key.publicKeyHex,
  };
  return signManifestEd25519(withPub as unknown as Record<string, unknown>, key, signedAt) as unknown as SignedAgentCard;
}

export function verifyAgentCard(
  signedCard: SignedAgentCard,
  trustedPublicKeyHex?: string,
): ManifestVerificationResult {
  if (!signedCard.id || !signedCard.name || !Array.isArray(signedCard.capabilities)) {
    return { valid: false, reason: 'invalid agent card schema', payloadSha512: '' };
  }
  const pinned = trustedPublicKeyHex ?? signedCard.publicKeyHex;
  return verifyManifestEd25519(signedCard as unknown as Record<string, unknown>, pinned);
}

/**
 * Cryptographic registry that rejects tampered or untrusted manifests and Agent Cards on registration and lookup.
 */
export class VerifiedArtifactRegistry {
  private readonly manifests = new Map<string, Record<string, unknown>>();
  private readonly agentCards = new Map<string, SignedAgentCard>();
  private readonly trustedKeys = new Set<string>();

  constructor(trustedPublicKeyHexes: readonly string[] = []) {
    for (const k of trustedPublicKeyHexes) {
      this.trustedKeys.add(k.toLowerCase());
    }
  }

  public addTrustedKey(publicKeyHex: string): void {
    this.trustedKeys.add(publicKeyHex.toLowerCase());
  }

  public registerManifest(id: string, signedManifest: Record<string, unknown>): ManifestVerificationResult {
    const sig = signedManifest.signature as VerifiedBootSignature | undefined;
    if (this.trustedKeys.size > 0 && (!sig || !this.trustedKeys.has(sig.publicKeyHex.toLowerCase()))) {
      return { valid: false, reason: 'untrusted signer public key', payloadSha512: '' };
    }
    const res = verifyManifestEd25519(signedManifest);
    if (res.valid) {
      this.manifests.set(id, signedManifest);
    }
    return res;
  }

  public getVerifiedManifest(id: string): Record<string, unknown> | null {
    const m = this.manifests.get(id);
    if (!m) return null;
    return verifyManifestEd25519(m).valid ? m : null;
  }

  public registerAgentCard(card: SignedAgentCard): ManifestVerificationResult {
    if (this.trustedKeys.size > 0 && !this.trustedKeys.has(card.signature.publicKeyHex.toLowerCase())) {
      return { valid: false, reason: 'untrusted agent card signer', payloadSha512: '' };
    }
    const res = verifyAgentCard(card);
    if (res.valid) {
      this.agentCards.set(card.id, card);
    }
    return res;
  }

  public getVerifiedAgentCard(id: string): SignedAgentCard | null {
    const card = this.agentCards.get(id);
    if (!card) return null;
    return verifyAgentCard(card).valid ? card : null;
  }
}
