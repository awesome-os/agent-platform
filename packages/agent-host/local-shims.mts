import { existsSync } from 'node:fs';
import { arch, cpus, hostname, platform, totalmem } from 'node:os';

export const SECTIUM_FAVICON_DATA_URI =
  'data:image/svg+xml,' +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#0e1116"/><path d="M8 16l5-8h6l5 8-5 8h-6z" fill="#39c5cf"/></svg>',
  );

export function getSectiumDevicePipePath(): string {
  return process.platform === 'win32' ? '\\\\.\\pipe\\sectium-dev' : '/tmp/sectium-dev.sock';
}

export function getHostEnvironmentSnapshot(): {
  platform: string;
  arch: string;
  hostname: string;
  cpuCount: number;
  totalRamGb: number;
  pipePath: string;
} {
  return {
    platform: platform(),
    arch: arch(),
    hostname: hostname(),
    cpuCount: cpus().length,
    totalRamGb: Math.round((totalmem() / 1024 ** 3) * 10) / 10,
    pipePath: getSectiumDevicePipePath(),
  };
}

export class SectiumDeviceClient {
  readonly endpoint: string;

  constructor(endpoint: string = getSectiumDevicePipePath()) {
    this.endpoint = endpoint;
  }

  isAvailable(): boolean {
    return process.platform !== 'win32' && existsSync(this.endpoint);
  }

  async ping(): Promise<{ connected: boolean; endpoint: string }> {
    return { connected: this.isAvailable(), endpoint: this.endpoint };
  }

  async sendCommand(_cmd: Record<string, unknown>): Promise<{ ok: boolean }> {
    return { ok: true };
  }
}
