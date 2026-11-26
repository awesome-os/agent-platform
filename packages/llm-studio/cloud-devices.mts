/**
 * @file cloud-devices.mts
 * @description Sovereign Unified Device Management, Cross-Boundary Registration,
 * Cryptographic Unified Device-ID Generation, and Consistent-Hash Sharding Ring
 * (`CloudDeviceMesh`) for the Hyperscaler Cloud Console.
 *
 * Zero external dependencies — pure Node `node:crypto` + `node:os`, `erasableSyntaxOnly: true`.
 */
import { createHash, createHmac, generateKeyPairSync, randomBytes } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import * as os from 'node:os';
import { dirname } from 'node:path';

export type DeviceTransport = 'os-pipe' | 'lan-mtls' | 'quic-tunnel' | 'webrtc-p2p' | 'ssh-bastion';
export type DeviceAuthMode = 'local-pipe-acl' | 'ed25519-mtls' | 'pairing-token' | 'oauth-oidc';
export type DeviceLocality = 'local-host' | 'lan-peer' | 'sovereign-edge' | 'public-cloud';
export type DeviceStatus = 'online' | 'offline' | 'pending' | 'revoked';

export interface DeviceHardwareSpec {
  hostname: string;
  platform: string;
  arch: string;
  cpuCores: number;
  ramMb: number;
  gpuModel?: string;
}

export interface DeviceRecord {
  deviceId: string;
  tenantId: string;
  name: string;
  transport: DeviceTransport;
  authMode: DeviceAuthMode;
  locality: DeviceLocality;
  endpoint: string;
  publicKeyFingerprint: string;
  shardSlot: number; // 0..1023 consistent-hash ring slot
  shardWeight: number;
  hardware: DeviceHardwareSpec;
  status: DeviceStatus;
  enrolledAt: string;
  lastHeartbeatAt: string;
}

export interface EnrollmentTicket {
  code: string;
  tenantId: string;
  transport: DeviceTransport;
  authMode: DeviceAuthMode;
  locality: DeviceLocality;
  signature: string;
  expiresAt: string;
  used: boolean;
}

export interface ShardRouteResult {
  key: string;
  ringPosition: number;
  primary: DeviceRecord | null;
  replicas: DeviceRecord[];
}

export interface CloudDeviceMeshOptions {
  persistPath?: string;
  org?: string;
  autoRegisterLocalHost?: boolean;
}

const RING_SIZE = 1024;

export function generateUnifiedDeviceId(input: {
  org?: string;
  locality: DeviceLocality;
  hostname: string;
  arch: string;
  platform: string;
  publicKeyFingerprint: string;
}): string {
  const locTag =
    input.locality === 'local-host'
      ? 'host'
      : input.locality === 'lan-peer'
        ? 'lan'
        : input.locality === 'sovereign-edge'
          ? 'edge'
          : 'cloud';

  const material = [
    input.org ?? 'awesome-os',
    input.locality,
    input.hostname.toLowerCase(),
    input.platform,
    input.arch,
    input.publicKeyFingerprint,
  ].join('|');

  const hashHex = createHash('sha256').update(material).digest('hex').slice(0, 16);
  return `b8g-dev-${locTag}-${hashHex}`;
}

export function computeRingSlot(key: string): number {
  const digest = createHash('sha256').update(key).digest();
  const u32 = digest.readUInt32BE(0);
  return u32 % RING_SIZE;
}

export class CloudDeviceMesh {
  readonly #org: string;
  readonly #persistPath: string | null;
  readonly #hmacSecret: Buffer;
  readonly #devices = new Map<string, DeviceRecord>();
  readonly #tickets = new Map<string, EnrollmentTicket>();

  constructor(opts: CloudDeviceMeshOptions = {}) {
    this.#org = opts.org ?? 'awesome-os';
    this.#persistPath = opts.persistPath ?? null;
    this.#hmacSecret = createHash('sha256')
      .update(`b8g-mesh-secret:${this.#org}:${os.hostname()}`)
      .digest();

    this.#loadFromDisk();

    if (opts.autoRegisterLocalHost !== false && this.#devices.size === 0) {
      this.registerLocalHostNode('root-host');
    }
  }

  registerLocalHostNode(tenantId = 'root-host'): DeviceRecord {
    const hw: DeviceHardwareSpec = {
      hostname: os.hostname(),
      platform: process.platform,
      arch: process.arch,
      cpuCores: Math.max(1, os.cpus().length),
      ramMb: Math.round(os.totalmem() / (1024 * 1024)),
    };
    const fp = createHash('sha256')
      .update(`local-host:${hw.hostname}:${hw.platform}:${hw.arch}`)
      .digest('hex')
      .slice(0, 32);

    return this.registerDevice({
      tenantId,
      name: `${hw.hostname} (Local Sovereign Host)`,
      transport: 'os-pipe',
      authMode: 'local-pipe-acl',
      locality: 'local-host',
      endpoint: process.platform === 'win32' ? '\\\\.\\pipe\\b8g-host-bus' : '/tmp/b8g-host-bus.sock',
      publicKeyFingerprint: fp,
      hardware: hw,
    });
  }

  registerDevice(input: {
    tenantId: string;
    name: string;
    transport: DeviceTransport;
    authMode: DeviceAuthMode;
    locality: DeviceLocality;
    endpoint: string;
    hardware: DeviceHardwareSpec;
    publicKeyFingerprint?: string;
    shardWeight?: number;
  }): DeviceRecord {
    let fp = input.publicKeyFingerprint;
    if (!fp) {
      const { publicKey } = generateKeyPairSync('ed25519');
      const der = publicKey.export({ type: 'spki', format: 'der' });
      fp = createHash('sha256').update(der).digest('hex').slice(0, 32);
    }

    const deviceId = generateUnifiedDeviceId({
      org: this.#org,
      locality: input.locality,
      hostname: input.hardware.hostname,
      arch: input.hardware.arch,
      platform: input.hardware.platform,
      publicKeyFingerprint: fp,
    });

    const now = new Date().toISOString();
    const record: DeviceRecord = {
      deviceId,
      tenantId: input.tenantId,
      name: input.name,
      transport: input.transport,
      authMode: input.authMode,
      locality: input.locality,
      endpoint: input.endpoint,
      publicKeyFingerprint: fp,
      shardSlot: computeRingSlot(deviceId),
      shardWeight: Math.max(1, input.shardWeight ?? input.hardware.cpuCores),
      hardware: { ...input.hardware },
      status: 'online',
      enrolledAt: now,
      lastHeartbeatAt: now,
    };

    this.#devices.set(deviceId, record);
    this.#saveToDisk();
    return { ...record, hardware: { ...record.hardware } };
  }

  issueEnrollmentTicket(
    tenantId: string,
    opts: {
      transport?: DeviceTransport;
      authMode?: DeviceAuthMode;
      locality?: DeviceLocality;
      ttlSec?: number;
    } = {},
  ): EnrollmentTicket {
    const transport: DeviceTransport = opts.transport ?? 'lan-mtls';
    const authMode: DeviceAuthMode = opts.authMode ?? 'ed25519-mtls';
    const locality: DeviceLocality = opts.locality ?? 'lan-peer';
    const ttlMs = Math.max(30, opts.ttlSec ?? 600) * 1000;
    const rawRand = randomBytes(4).toString('hex').toUpperCase();
    const code = `B8G-${rawRand.slice(0, 4)}-${rawRand.slice(4, 8)}`;
    const expiresAt = new Date(Date.now() + ttlMs).toISOString();
    const sig = createHmac('sha256', this.#hmacSecret)
      .update(`${code}|${tenantId}|${transport}|${authMode}|${locality}|${expiresAt}`)
      .digest('hex')
      .slice(0, 24);

    const ticket: EnrollmentTicket = {
      code,
      tenantId,
      transport,
      authMode,
      locality,
      signature: sig,
      expiresAt,
      used: false,
    };
    this.#tickets.set(code, ticket);
    this.#saveToDisk();
    return { ...ticket };
  }

  enrollWithTicket(
    code: string,
    spec: {
      name: string;
      endpoint: string;
      hardware: DeviceHardwareSpec;
      publicKeyFingerprint?: string;
    },
  ): { ok: boolean; device?: DeviceRecord; error?: string } {
    const ticket = this.#tickets.get(code.trim().toUpperCase());
    if (!ticket) return { ok: false, error: `Unknown enrollment ticket: ${code}` };
    if (ticket.used) return { ok: false, error: `Enrollment ticket already used: ${code}` };
    if (Date.now() > Date.parse(ticket.expiresAt)) {
      return { ok: false, error: `Enrollment ticket expired: ${code}` };
    }

    const expectedSig = createHmac('sha256', this.#hmacSecret)
      .update(`${ticket.code}|${ticket.tenantId}|${ticket.transport}|${ticket.authMode}|${ticket.locality}|${ticket.expiresAt}`)
      .digest('hex')
      .slice(0, 24);
    if (expectedSig !== ticket.signature) {
      return { ok: false, error: 'Ticket HMAC signature verification failed' };
    }

    ticket.used = true;
    const device = this.registerDevice({
      tenantId: ticket.tenantId,
      name: spec.name,
      transport: ticket.transport,
      authMode: ticket.authMode,
      locality: ticket.locality,
      endpoint: spec.endpoint,
      hardware: spec.hardware,
      publicKeyFingerprint: spec.publicKeyFingerprint,
    });
    return { ok: true, device };
  }

  listDevices(tenantId?: string): DeviceRecord[] {
    const out: DeviceRecord[] = [];
    for (const d of this.#devices.values()) {
      if (!tenantId || d.tenantId === tenantId) {
        out.push({ ...d, hardware: { ...d.hardware } });
      }
    }
    return out.sort((a, b) => a.shardSlot - b.shardSlot);
  }

  listTickets(tenantId?: string): EnrollmentTicket[] {
    const out: EnrollmentTicket[] = [];
    for (const t of this.#tickets.values()) {
      if (!tenantId || t.tenantId === tenantId) out.push({ ...t });
    }
    return out;
  }

  revokeDevice(deviceId: string): boolean {
    const d = this.#devices.get(deviceId);
    if (!d) return false;
    d.status = 'revoked';
    this.#saveToDisk();
    return true;
  }

  heartbeat(deviceId: string): boolean {
    const d = this.#devices.get(deviceId);
    if (!d || d.status === 'revoked') return false;
    d.status = 'online';
    d.lastHeartbeatAt = new Date().toISOString();
    return true;
  }

  resolveShardOwner(key: string, tenantId?: string, replicaCount = 2): ShardRouteResult {
    const ringPosition = computeRingSlot(key);
    const active = this.listDevices(tenantId).filter((d) => d.status === 'online');
    if (active.length === 0) {
      return { key, ringPosition, primary: null, replicas: [] };
    }

    // Consistent-hash clockwise walk on 0..1023 ring
    const startIdx = active.findIndex((d) => d.shardSlot >= ringPosition);
    const ordered: DeviceRecord[] = [];
    const base = startIdx >= 0 ? startIdx : 0;
    for (let i = 0; i < active.length; i++) {
      ordered.push(active[(base + i) % active.length]!);
    }

    return {
      key,
      ringPosition,
      primary: ordered[0] ?? null,
      replicas: ordered.slice(1, Math.max(1, replicaCount)),
    };
  }

  #loadFromDisk(): void {
    if (!this.#persistPath || !existsSync(this.#persistPath)) return;
    try {
      const raw = JSON.parse(readFileSync(this.#persistPath, 'utf8')) as {
        devices?: DeviceRecord[];
        tickets?: EnrollmentTicket[];
      };
      for (const d of raw.devices ?? []) this.#devices.set(d.deviceId, d);
      for (const t of raw.tickets ?? []) this.#tickets.set(t.code, t);
    } catch {
      // Ignore corrupt persistence file
    }
  }

  #saveToDisk(): void {
    if (!this.#persistPath) return;
    try {
      mkdirSync(dirname(this.#persistPath), { recursive: true });
      const tmp = `${this.#persistPath}.tmp`;
      writeFileSync(
        tmp,
        JSON.stringify(
          {
            devices: [...this.#devices.values()],
            tickets: [...this.#tickets.values()],
          },
          null,
          2,
        ),
        'utf8',
      );
      renameSync(tmp, this.#persistPath);
    } catch {
      // Non-fatal persistence error
    }
  }
}
