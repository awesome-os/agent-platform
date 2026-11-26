/**
 * @file packages/browser-use/controller.mts
 * @description BrowserUseController supporting `navigate`, `click`, `type`, `scroll`,
 * `extract`, and `screenshot` over an injectable CDP / `sectium-device` pipe transport
 * interface (`BrowserTransport`) plus a deterministic `MockBrowserTransport` for
 * hermetic testing. Zero external browser driver bloat, zero ad-hoc binary spawning.
 *
 * Strictly `erasableSyntaxOnly: true`, Node 22+ built-ins only.
 */

import { createHash, randomBytes } from 'node:crypto';
import { createConnection } from 'node:net';
import { performance } from 'node:perf_hooks';
import {
  DOMTreeSerializer,
  type DomNodeSnapshot,
  type DomSerializerOptions,
  type IndexedDomElement,
  type SerializedDomMap,
} from './dom-serializer.mts';

// ============================================================================
// 1. Transport Interfaces (CDP / sectium-device Pipe)
// ============================================================================

export type BrowserCommandMethod =
  | 'Page.navigate'
  | 'Page.captureScreenshot'
  | 'DOM.getSnapshot'
  | 'DOM.click'
  | 'DOM.type'
  | 'Input.scroll'
  | 'Runtime.extractText'
  | 'sectium.attach_channel'
  | 'sectium.open_window'
  | 'sectium.ping';

export interface BrowserTransportRequest {
  readonly id: string;
  readonly method: BrowserCommandMethod | string;
  readonly params?: Readonly<Record<string, unknown>>;
}

export interface BrowserTransportResponse<T = unknown> {
  readonly id: string;
  readonly ok: boolean;
  readonly result?: T;
  readonly error?: string;
  readonly code?: string;
}

export interface BrowserTransport {
  readonly kind: 'cdp' | 'sectium-device' | 'mock';
  send<T = unknown>(
    method: BrowserCommandMethod | string,
    params?: Readonly<Record<string, unknown>>,
  ): Promise<T>;
  close?(): Promise<void> | void;
}

export interface SectiumPipeTransportOptions {
  readonly pipePath?: string;
  readonly clientId?: string;
  readonly timeoutMs?: number;
}

export function getDefaultSectiumPipePath(override?: string): string {
  const trimmed = override?.trim();
  if (trimmed) {
    return trimmed;
  }
  return process.platform === 'win32' ? '\\\\.\\pipe\\sectium-dev' : '/tmp/sectium-dev.sock';
}

export class SectiumDeviceOfflineError extends Error {
  public readonly code = 'DEVICE_OFFLINE';
  public readonly pipePath: string;

  constructor(pipePath: string, causeMessage?: string) {
    super(
      `Sectium device offline on ${pipePath}${causeMessage ? ` (${causeMessage})` : ''}`,
    );
    this.name = 'SectiumDeviceOfflineError';
    this.pipePath = pipePath;
  }
}

/**
 * Network-device NDJSON transport communicating with `sectium-device` over
 * `\\.\pipe\sectium-dev` (Win32) or `/tmp/sectium-dev.sock` (POSIX) on
 * `channel: 'command'`. Never spawns a browser binary directly.
 */
export class SectiumPipeTransport implements BrowserTransport {
  public readonly kind = 'sectium-device' as const;
  public readonly pipePath: string;
  public readonly clientId: string;
  private readonly timeoutMs: number;
  private seq = 0;

  constructor(options: SectiumPipeTransportOptions = {}) {
    this.pipePath = getDefaultSectiumPipePath(options.pipePath);
    this.clientId = options.clientId ?? `browser-use-${process.pid}`;
    this.timeoutMs = Math.max(50, options.timeoutMs ?? 1500);
  }

  public async attachCommandChannel(): Promise<Record<string, unknown>> {
    return this.send<Record<string, unknown>>('sectium.attach_channel', {
      op: 'attach_channel',
      channel: 'command',
      clientId: this.clientId,
      sink: 'cdp-bridge',
    });
  }

  public async send<T = unknown>(
    method: BrowserCommandMethod | string,
    params: Readonly<Record<string, unknown>> = {},
  ): Promise<T> {
    this.seq += 1;
    const id = `bu-${this.seq}-${randomBytes(3).toString('hex')}`;
    const payload =
      method.startsWith('sectium.')
        ? { id, ...params, op: params.op ?? method.slice('sectium.'.length) }
        : { id, channel: 'command', op: 'cdp', method, params };

    return new Promise<T>((resolve, reject) => {
      const socket = createConnection(this.pipePath);
      let settled = false;
      let buffer = '';

      const timer = setTimeout(() => {
        if (!settled) {
          settled = true;
          socket.destroy();
          reject(new Error(`Sectium command timeout after ${this.timeoutMs} ms (${method})`));
        }
      }, this.timeoutMs);

      const finish = (err: Error | null, value?: T): void => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(timer);
        socket.destroy();
        if (err) {
          reject(err);
        } else {
          resolve(value as T);
        }
      };

      socket.on('connect', () => {
        socket.write(`${JSON.stringify(payload)}\n`, 'utf8');
      });

      socket.on('data', (chunk: Buffer) => {
        buffer += chunk.toString('utf8');
        const nlIdx = buffer.indexOf('\n');
        if (nlIdx === -1) {
          return;
        }
        const line = buffer.slice(0, nlIdx).trim();
        try {
          const parsed = JSON.parse(line) as {
            readonly ok?: boolean;
            readonly error?: string;
            readonly result?: T;
          };
          if (parsed.ok === false) {
            finish(new Error(parsed.error ?? `Sectium command failed: ${method}`));
            return;
          }
          finish(null, (parsed.result !== undefined ? parsed.result : parsed) as T);
        } catch (parseErr) {
          finish(parseErr instanceof Error ? parseErr : new Error(String(parseErr)));
        }
      });

      socket.on('error', (err: NodeJS.ErrnoException) => {
        const offlineCodes = new Set(['ENOENT', 'ECONNREFUSED', 'ECONNRESET', 'EPIPE', 'EACCES']);
        if (err.code && offlineCodes.has(err.code)) {
          finish(new SectiumDeviceOfflineError(this.pipePath, err.code));
          return;
        }
        finish(err);
      });
    });
  }
}

// ============================================================================
// 2. Deterministic MockBrowserTransport (Hermetic Testing)
// ============================================================================

export interface MockPageState {
  readonly url: string;
  readonly title: string;
  readonly dom: DomNodeSnapshot | string;
  readonly textContent?: string;
}

export interface RecordedTransportCall {
  readonly seq: number;
  readonly method: string;
  readonly params: Readonly<Record<string, unknown>>;
}

/**
 * Deterministic in-memory browser transport for hermetic `browser-use` testing.
 * Records every CDP/Sectium command and simulates stateful page navigation,
 * form input values, scroll offsets, text extraction, and screenshots.
 */
export class MockBrowserTransport implements BrowserTransport {
  public readonly kind = 'mock' as const;
  private readonly pages = new Map<string, MockPageState>();
  private readonly calls: RecordedTransportCall[] = [];
  private readonly inputValues = new Map<string, string>();
  private currentUrl = 'about:blank';
  private scrollY = 0;
  private scrollX = 0;
  private clickHook?: (target: string, page: MockPageState) => string | void;

  constructor(initialPages: readonly MockPageState[] = [], startUrl?: string) {
    for (const page of initialPages) {
      this.pages.set(page.url, page);
    }
    if (startUrl) {
      this.currentUrl = startUrl;
    } else if (initialPages.length > 0) {
      this.currentUrl = initialPages[0]!.url;
    }
  }

  public registerPage(page: MockPageState): void {
    this.pages.set(page.url, page);
  }

  public onClick(hook: (target: string, page: MockPageState) => string | void): void {
    this.clickHook = hook;
  }

  public getRecordedCalls(): readonly RecordedTransportCall[] {
    return Object.freeze([...this.calls]);
  }

  public getInputValue(selector: string): string | undefined {
    return this.inputValues.get(selector);
  }

  public getScrollPosition(): { readonly x: number; readonly y: number } {
    return Object.freeze({ x: this.scrollX, y: this.scrollY });
  }

  private getCurrentPage(): MockPageState {
    return (
      this.pages.get(this.currentUrl) ?? {
        url: this.currentUrl,
        title: 'Blank Page',
        dom: { tag: 'document', children: [] },
        textContent: '',
      }
    );
  }

  public async send<T = unknown>(
    method: BrowserCommandMethod | string,
    params: Readonly<Record<string, unknown>> = {},
  ): Promise<T> {
    this.calls.push(
      Object.freeze({
        seq: this.calls.length + 1,
        method,
        params: Object.freeze({ ...params }),
      }),
    );

    switch (method) {
      case 'Page.navigate': {
        const url = String(params.url ?? 'about:blank');
        this.currentUrl = url;
        this.scrollX = 0;
        this.scrollY = 0;
        const page = this.getCurrentPage();
        return { url: page.url, title: page.title } as T;
      }
      case 'DOM.getSnapshot': {
        const page = this.getCurrentPage();
        return { url: page.url, title: page.title, dom: page.dom } as T;
      }
      case 'DOM.click': {
        const selector = String(params.selector ?? '');
        const page = this.getCurrentPage();
        if (this.clickHook) {
          const nextUrl = this.clickHook(selector, page);
          if (typeof nextUrl === 'string' && nextUrl.length > 0) {
            this.currentUrl = nextUrl;
          }
        } else if (typeof params.href === 'string' && params.href.length > 0) {
          this.currentUrl = params.href;
        }
        return { clicked: true, selector, url: this.currentUrl } as T;
      }
      case 'DOM.type': {
        const selector = String(params.selector ?? '');
        const text = String(params.text ?? '');
        const clear = params.clear !== false;
        const prev = clear ? '' : (this.inputValues.get(selector) ?? '');
        const nextVal = `${prev}${text}`;
        this.inputValues.set(selector, nextVal);
        return { typed: true, selector, length: nextVal.length } as T;
      }
      case 'Input.scroll': {
        const dx = Number(params.deltaX ?? 0);
        const dy = Number(params.deltaY ?? 0);
        if (params.toTop === true) {
          this.scrollX = 0;
          this.scrollY = 0;
        } else if (params.toBottom === true) {
          this.scrollY = 10_000;
        } else {
          this.scrollX = Math.max(0, this.scrollX + dx);
          this.scrollY = Math.max(0, this.scrollY + dy);
        }
        return { x: this.scrollX, y: this.scrollY } as T;
      }
      case 'Runtime.extractText': {
        const page = this.getCurrentPage();
        const raw =
          page.textContent ??
          (typeof page.dom === 'string'
            ? page.dom.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim()
            : JSON.stringify(page.dom));
        const maxChars = typeof params.maxChars === 'number' ? params.maxChars : 4000;
        return { url: page.url, title: page.title, text: raw.slice(0, maxChars) } as T;
      }
      case 'Page.captureScreenshot': {
        const page = this.getCurrentPage();
        const payload = `MOCK_PNG:${page.url}:${this.scrollY}:${page.title}`;
        const dataBase64 = Buffer.from(payload, 'utf8').toString('base64');
        return {
          dataBase64,
          mimeType: 'image/png',
          bytes: Buffer.byteLength(payload, 'utf8'),
        } as T;
      }
      default:
        return { ok: true, method } as T;
    }
  }
}

// ============================================================================
// 3. BrowserUseController
// ============================================================================

export type BrowserActionName =
  | 'navigate'
  | 'click'
  | 'type'
  | 'scroll'
  | 'extract'
  | 'screenshot'
  | 'observe';

export interface BrowserActionResult<T = unknown> {
  readonly ok: boolean;
  readonly action: BrowserActionName;
  readonly url: string;
  readonly latencyMs: number;
  readonly target?: string;
  readonly data?: T;
  readonly error?: string;
}

export interface BrowserObservation {
  readonly url: string;
  readonly title: string;
  readonly domMap: SerializedDomMap;
  readonly domHash: string;
}

export interface ScreenshotCapture {
  readonly url: string;
  readonly mimeType: 'image/png' | 'image/jpeg';
  readonly dataBase64: string;
  readonly bytes: number;
  readonly sha512: string;
}

export class BrowserUseController {
  public readonly transport: BrowserTransport;
  public readonly serializer: DOMTreeSerializer;
  private lastObservation: BrowserObservation | null = null;
  private currentUrl = 'about:blank';

  constructor(transport: BrowserTransport, serializer: DOMTreeSerializer = new DOMTreeSerializer()) {
    this.transport = transport;
    this.serializer = serializer;
  }

  public getLastObservation(): BrowserObservation | null {
    return this.lastObservation;
  }

  public getCurrentUrl(): string {
    return this.currentUrl;
  }

  private resolveTarget(target: number | string): {
    readonly selector: string;
    readonly element?: IndexedDomElement;
  } {
    if (typeof target === 'number') {
      const el = this.lastObservation?.domMap.selectorMap.get(target);
      if (!el) {
        throw new Error(`Element index [${target}] not found in last DOM observation`);
      }
      return { selector: el.selector, element: el };
    }
    return { selector: target };
  }

  public async observe(options?: DomSerializerOptions): Promise<BrowserObservation> {
    const snap = await this.transport.send<{
      readonly url?: string;
      readonly title?: string;
      readonly dom: DomNodeSnapshot | string;
    }>('DOM.getSnapshot', {});
    const url = snap.url ?? this.currentUrl;
    this.currentUrl = url;
    const domMap = this.serializer.serialize(snap.dom, options);
    const domHash = createHash('sha512').update(domMap.text, 'utf8').digest('hex');
    const obs: BrowserObservation = Object.freeze({
      url,
      title: snap.title ?? '',
      domMap,
      domHash,
    });
    this.lastObservation = obs;
    return obs;
  }

  public async navigate(url: string): Promise<BrowserActionResult<{ readonly title: string }>> {
    const t0 = performance.now();
    const res = await this.transport.send<{ readonly url?: string; readonly title?: string }>(
      'Page.navigate',
      { url },
    );
    this.currentUrl = res.url ?? url;
    this.lastObservation = null;
    return Object.freeze({
      ok: true,
      action: 'navigate',
      url: this.currentUrl,
      target: url,
      latencyMs: Math.max(0, performance.now() - t0),
      data: Object.freeze({ title: res.title ?? '' }),
    });
  }

  public async click(
    target: number | string,
  ): Promise<BrowserActionResult<{ readonly selector: string }>> {
    const t0 = performance.now();
    const resolved = this.resolveTarget(target);
    const res = await this.transport.send<{ readonly url?: string }>(
      'DOM.click',
      {
        selector: resolved.selector,
        ...(resolved.element?.backendNodeId !== undefined
          ? { backendNodeId: resolved.element.backendNodeId }
          : {}),
        ...(resolved.element?.href !== undefined ? { href: resolved.element.href } : {}),
      },
    );
    if (res.url) {
      this.currentUrl = res.url;
    }
    return Object.freeze({
      ok: true,
      action: 'click',
      url: this.currentUrl,
      target: String(target),
      latencyMs: Math.max(0, performance.now() - t0),
      data: Object.freeze({ selector: resolved.selector }),
    });
  }

  public async type(
    target: number | string,
    text: string,
    opts?: { readonly clear?: boolean; readonly submit?: boolean },
  ): Promise<BrowserActionResult<{ readonly selector: string; readonly chars: number }>> {
    const t0 = performance.now();
    const resolved = this.resolveTarget(target);
    await this.transport.send('DOM.type', {
      selector: resolved.selector,
      text,
      clear: opts?.clear ?? true,
      submit: opts?.submit ?? false,
      ...(resolved.element?.backendNodeId !== undefined
        ? { backendNodeId: resolved.element.backendNodeId }
        : {}),
    });
    return Object.freeze({
      ok: true,
      action: 'type',
      url: this.currentUrl,
      target: String(target),
      latencyMs: Math.max(0, performance.now() - t0),
      data: Object.freeze({ selector: resolved.selector, chars: text.length }),
    });
  }

  public async scroll(
    direction: 'up' | 'down' | 'top' | 'bottom' | { readonly deltaX?: number; readonly deltaY: number },
    amountPx = 400,
  ): Promise<BrowserActionResult<{ readonly x: number; readonly y: number }>> {
    const t0 = performance.now();
    let params: Record<string, unknown>;
    if (typeof direction === 'string') {
      if (direction === 'up') {
        params = { deltaX: 0, deltaY: -Math.abs(amountPx) };
      } else if (direction === 'down') {
        params = { deltaX: 0, deltaY: Math.abs(amountPx) };
      } else if (direction === 'top') {
        params = { toTop: true, deltaX: 0, deltaY: 0 };
      } else {
        params = { toBottom: true, deltaX: 0, deltaY: 0 };
      }
    } else {
      params = { deltaX: direction.deltaX ?? 0, deltaY: direction.deltaY };
    }
    const pos = await this.transport.send<{ readonly x?: number; readonly y?: number }>(
      'Input.scroll',
      params,
    );
    return Object.freeze({
      ok: true,
      action: 'scroll',
      url: this.currentUrl,
      target: typeof direction === 'string' ? direction : `dy:${direction.deltaY}`,
      latencyMs: Math.max(0, performance.now() - t0),
      data: Object.freeze({ x: pos.x ?? 0, y: pos.y ?? 0 }),
    });
  }

  public async extract(opts?: {
    readonly selector?: string;
    readonly maxChars?: number;
    readonly format?: 'text' | 'dom-map';
  }): Promise<BrowserActionResult<{ readonly content: string; readonly format: string }>> {
    const t0 = performance.now();
    if (opts?.format === 'dom-map') {
      const obs = await this.observe();
      return Object.freeze({
        ok: true,
        action: 'extract',
        url: obs.url,
        target: opts.selector ?? 'document',
        latencyMs: Math.max(0, performance.now() - t0),
        data: Object.freeze({
          content: obs.domMap.text.slice(0, opts.maxChars ?? 8000),
          format: 'dom-map',
        }),
      });
    }
    const res = await this.transport.send<{ readonly url?: string; readonly text?: string }>(
      'Runtime.extractText',
      {
        ...(opts?.selector !== undefined ? { selector: opts.selector } : {}),
        maxChars: opts?.maxChars ?? 4000,
      },
    );
    return Object.freeze({
      ok: true,
      action: 'extract',
      url: res.url ?? this.currentUrl,
      target: opts?.selector ?? 'document',
      latencyMs: Math.max(0, performance.now() - t0),
      data: Object.freeze({ content: res.text ?? '', format: 'text' }),
    });
  }

  public async screenshot(opts?: {
    readonly fullPage?: boolean;
    readonly format?: 'png' | 'jpeg';
  }): Promise<BrowserActionResult<ScreenshotCapture>> {
    const t0 = performance.now();
    const format = opts?.format ?? 'png';
    const res = await this.transport.send<{
      readonly dataBase64: string;
      readonly mimeType?: 'image/png' | 'image/jpeg';
      readonly bytes?: number;
    }>('Page.captureScreenshot', {
      fullPage: opts?.fullPage ?? false,
      format,
    });
    const rawBuf = Buffer.from(res.dataBase64, 'base64');
    const sha512 = createHash('sha512').update(rawBuf).digest('hex');
    const capture: ScreenshotCapture = Object.freeze({
      url: this.currentUrl,
      mimeType: res.mimeType ?? (format === 'jpeg' ? 'image/jpeg' : 'image/png'),
      dataBase64: res.dataBase64,
      bytes: res.bytes ?? rawBuf.byteLength,
      sha512,
    });
    return Object.freeze({
      ok: true,
      action: 'screenshot',
      url: this.currentUrl,
      latencyMs: Math.max(0, performance.now() - t0),
      data: capture,
    });
  }
}
