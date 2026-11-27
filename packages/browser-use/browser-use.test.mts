import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as net from 'node:net';
import * as os from 'node:os';
import * as path from 'node:path';
import { describe, it } from 'node:test';
import {
  SubstrateCasStore,
  SubstrateKvStore,
  Zer0qRunner,
} from '../zer0q-substrate/index.mts';
import {
  BrowserUseAgent,
  BrowserUseController,
  DOMTreeSerializer,
  MockBrowserTransport,
  SectiumDeviceOfflineError,
  SectiumPipeTransport,
  type DomNodeSnapshot,
} from './index.mts';

describe('packages/browser-use — DOMTreeSerializer, Controller, Sectium Pipe & Agent', () => {
  it('DOMTreeSerializer produces indexed [idx] <tag role="...">"text" lines and selectorMap', () => {
    const dom: DomNodeSnapshot = {
      tag: 'main',
      children: [
        { tag: 'a', href: '/home', text: 'Home' },
        { tag: 'input', type: 'text', name: 'query', placeholder: 'Search docs' },
        { tag: 'button', type: 'submit', text: 'Run Search' },
        { tag: 'div', role: 'tab', text: 'Telemetry' },
      ],
    };

    const serializer = new DOMTreeSerializer({ maxTokens: 500 });
    const mapped = serializer.serialize(dom);

    assert.equal(mapped.elements.length, 4);
    assert.equal(mapped.truncated, false);
    assert.match(mapped.lines[0]!, /^\[1\] <a role="link" href="\/home">"Home"$/);
    assert.match(mapped.lines[1]!, /^\[2\] <input role="textbox" type="text" name="query">"Search docs"$/);
    assert.match(mapped.lines[2]!, /^\[3\] <button role="button" type="submit">"Run Search"$/);
    assert.match(mapped.lines[3]!, /^\[4\] <div role="tab">"Telemetry"$/);
    assert.equal(mapped.selectorMap.get(2)?.selector, 'input[name="query"]');
  });

  it('DOMTreeSerializer filters hidden, styled-invisible, zero-bounds, viewport-clipped, and hidden-subtree nodes', () => {
    const dom: DomNodeSnapshot = {
      tag: 'body',
      children: [
        { tag: 'button', text: 'Visible', bounds: { x: 10, y: 10, width: 80, height: 30 } },
        { tag: 'button', text: 'HiddenAttr', hidden: true },
        { tag: 'button', text: 'ExplicitInvisible', visible: false },
        { tag: 'button', text: 'DisplayNone', style: 'display: none; color: red' },
        { tag: 'button', text: 'OpacityZero', style: 'opacity: 0' },
        { tag: 'button', text: 'ZeroBounds', bounds: { x: 0, y: 0, width: 0, height: 24 } },
        { tag: 'button', text: 'Offscreen', bounds: { x: 2000, y: 10, width: 50, height: 20 } },
        {
          tag: 'div',
          style: 'visibility: hidden',
          children: [{ tag: 'button', text: 'NestedHidden' }],
        },
      ],
    };

    const serializer = new DOMTreeSerializer();
    const mapped = serializer.serialize(dom, { viewport: { width: 1280, height: 720 } });
    assert.equal(mapped.elements.length, 1);
    assert.equal(mapped.elements[0]?.text, 'Visible');
  });

  it('DOMTreeSerializer enforces bounded token budget and maxElements with omittedCount', () => {
    const children: DomNodeSnapshot[] = Array.from({ length: 20 }, (_, i) => ({
      tag: 'button',
      id: `btn-${i + 1}`,
      text: `Action Button Number ${i + 1}`,
    }));
    const serializer = new DOMTreeSerializer({ maxTokens: 35 });
    const tight = serializer.serialize({ tag: 'div', children });

    assert.equal(tight.truncated, true);
    assert.ok(tight.estimatedTokens <= 35);
    assert.equal(tight.totalInteractiveVisible, 20);
    assert.equal(tight.omittedCount, 20 - tight.elements.length);

    const cappedByCount = serializer.serialize({ tag: 'div', children }, { maxTokens: 5000, maxElements: 3 });
    assert.equal(cappedByCount.truncated, true);
    assert.equal(cappedByCount.elements.length, 3);
    assert.equal(cappedByCount.omittedCount, 17);
  });

  it('DOMTreeSerializer parses raw HTML strings and respects includeDisabled option', () => {
    const html = `
      <div>
        <button id="go">Launch</button>
        <button aria-hidden="true">ScreenReaderHidden</button>
        <button disabled>DisabledBtn</button>
        <input type="checkbox" name="agree" aria-label="Accept terms" />
      </div>
    `;
    const serializer = new DOMTreeSerializer();
    const filtered = serializer.serialize(html);
    assert.equal(filtered.elements.length, 2);
    assert.equal(filtered.elements[0]?.selector, '#go');
    assert.equal(filtered.elements[1]?.role, 'checkbox');
    assert.equal(filtered.elements[1]?.text, 'Accept terms');

    const withDisabled = serializer.serialize(html, { includeDisabled: true });
    assert.equal(withDisabled.elements.length, 3);
    assert.match(withDisabled.lines[1]!, /disabled="true"/);
  });

  it('BrowserUseController executes navigate, observe, click, and type over MockBrowserTransport', async () => {
    const transport = new MockBrowserTransport([
      {
        url: 'https://awesome-os.local/login',
        title: 'AwesomeOS Login',
        dom: {
          tag: 'main',
          children: [
            { tag: 'input', id: 'user', type: 'text', placeholder: 'Username' },
            { tag: 'button', id: 'submit', text: 'Sign In' },
          ],
        },
        textContent: 'Welcome to AwesomeOS Login',
      },
      {
        url: 'https://awesome-os.local/dashboard',
        title: 'Dashboard',
        dom: {
          tag: 'main',
          children: [{ tag: 'a', id: 'metrics', href: '/metrics', text: 'View Metrics' }],
        },
        textContent: 'Agent Platform Dashboard Ready',
      },
    ]);
    transport.onClick((selector) => (selector === '#submit' ? 'https://awesome-os.local/dashboard' : undefined));

    const controller = new BrowserUseController(transport);
    const nav = await controller.navigate('https://awesome-os.local/login');
    assert.equal(nav.ok, true);
    assert.equal(nav.url, 'https://awesome-os.local/login');

    const obs = await controller.observe();
    assert.equal(obs.domMap.elements.length, 2);

    await controller.type(1, 'ali');
    await controller.type(1, 'ce', { clear: false });
    assert.equal(transport.getInputValue('#user'), 'alice');

    const clicked = await controller.click(2);
    assert.equal(clicked.ok, true);
    assert.equal(clicked.url, 'https://awesome-os.local/dashboard');
  });

  it('BrowserUseController executes scroll, extract (text/dom-map), and screenshot with SHA-512', async () => {
    const transport = new MockBrowserTransport([
      {
        url: 'https://awesome-os.local/report',
        title: 'Report',
        dom: {
          tag: 'main',
          children: [{ tag: 'button', id: 'export', text: 'Export CSV' }],
        },
        textContent: 'Telemetry Report Summary',
      },
    ]);
    const controller = new BrowserUseController(transport);

    const down = await controller.scroll('down', 320);
    assert.equal(down.data?.y, 320);
    const top = await controller.scroll('top');
    assert.equal(top.data?.y, 0);

    const textExtract = await controller.extract({ format: 'text' });
    assert.equal(textExtract.data?.content, 'Telemetry Report Summary');

    const mapExtract = await controller.extract({ format: 'dom-map' });
    assert.match(mapExtract.data?.content ?? '', /\[1\] <button role="button">"Export CSV"/);

    const shot = await controller.screenshot({ format: 'png' });
    assert.equal(shot.ok, true);
    assert.equal(shot.data?.mimeType, 'image/png');
    assert.equal(shot.data?.sha512.length, 128);
  });

  it('SectiumPipeTransport communicates NDJSON over channel:command and fails closed with DEVICE_OFFLINE when down', async () => {
    const missingPipe =
      process.platform === 'win32'
        ? `\\\\.\\pipe\\sectium-nonexistent-${process.pid}`
        : path.join(os.tmpdir(), `sectium-nonexistent-${process.pid}.sock`);
    const offlineTransport = new SectiumPipeTransport({ pipePath: missingPipe, timeoutMs: 300 });
    await assert.rejects(
      () => offlineTransport.attachCommandChannel(),
      (err: unknown) =>
        err instanceof SectiumDeviceOfflineError &&
        err.code === 'DEVICE_OFFLINE' &&
        err.pipePath === missingPipe,
    );

    // Live NDJSON pipe server round-trip
    const pipePath =
      process.platform === 'win32'
        ? `\\\\.\\pipe\\sectium-test-${process.pid}-${Date.now()}`
        : path.join(os.tmpdir(), `sectium-test-${process.pid}-${Date.now()}.sock`);
    const receivedFrames: Array<Record<string, unknown>> = [];
    const server = net.createServer((socket) => {
      let buf = '';
      socket.on('data', (chunk) => {
        buf += chunk.toString('utf8');
        const idx = buf.indexOf('\n');
        if (idx !== -1) {
          const req = JSON.parse(buf.slice(0, idx)) as Record<string, unknown>;
          receivedFrames.push(req);
          socket.write(`${JSON.stringify({ id: req.id, ok: true, result: { url: 'https://sectium.local/ok' } })}\n`);
        }
      });
    });

    await new Promise<void>((resolve) => server.listen(pipePath, resolve));
    try {
      const liveTransport = new SectiumPipeTransport({ pipePath, clientId: 'test-client', timeoutMs: 1000 });
      await liveTransport.attachCommandChannel();
      const navRes = await liveTransport.send<{ readonly url: string }>('Page.navigate', {
        url: 'https://sectium.local/ok',
      });
      assert.equal(navRes.url, 'https://sectium.local/ok');
      assert.equal(receivedFrames[0]?.op, 'attach_channel');
      assert.equal(receivedFrames[0]?.channel, 'command');
      assert.equal(receivedFrames[1]?.channel, 'command');
      assert.equal(receivedFrames[1]?.method, 'Page.navigate');
    } finally {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });

  it('BrowserUseAgent enforces bounded step history and emits secret-scrubbed NDJSON receipts', async () => {
    const transport = new MockBrowserTransport([
      {
        url: 'https://awesome-os.local/app',
        title: 'App',
        dom: { tag: 'main', children: [{ tag: 'button', id: 'next', text: 'Next' }] },
        textContent: 'App Content',
      },
    ]);
    const controller = new BrowserUseController(transport);
    const streamed: string[] = [];
    const agent = new BrowserUseAgent({
      controller,
      maxSteps: 6,
      maxHistorySteps: 2,
      onReceipt: (_r, line) => streamed.push(line),
    });

    const runRes = await agent.run({
      task: 'Scroll and finish',
      startUrl: 'https://awesome-os.local/app',
      planner: ({ step }) => {
        if (step <= 3) {
          return { action: 'scroll', direction: 'down', amountPx: 100, tokensUsed: 15 };
        }
        return {
          action: 'done',
          result: { summary: 'Done', leakedToken: 'ghp_1234567890abcdefghijklmnop' }, // holo-ci-allow-secret
          tokensUsed: 10,
        };
      },
    });

    assert.equal(runRes.status, 'COMPLETED');
    assert.equal(runRes.stepsExecuted, 4);
    assert.equal(runRes.history.length, 2);
    assert.equal(runRes.tokensUsed, 55);
    assert.ok(streamed.length >= 4);
    assert.equal(JSON.stringify(runRes.result).includes('ghp_1234567890'), false); // holo-ci-allow-secret
    for (const line of runRes.ndjson.split('\n')) {
      const parsed = JSON.parse(line) as { readonly seq: number; readonly inputDigest: string };
      assert.ok(parsed.seq >= 1);
      assert.equal(parsed.inputDigest.length, 128);
    }
  });

  it('BrowserUseAgent short-circuits repeated tasks via Zero-Q SKIMMER CAS hook with 0 tokens', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'browser-use-zer0q-'));
    try {
      const cas = new SubstrateCasStore(tmpDir);
      const kv = new SubstrateKvStore();
      const runner = new Zer0qRunner(cas, kv);

      const transport = new MockBrowserTransport([
        {
          url: 'https://awesome-os.local/search',
          title: 'Search',
          dom: {
            tag: 'main',
            children: [{ tag: 'input', id: 'q', type: 'search', placeholder: 'Query' }],
          },
          textContent: 'Holo v11 ready',
        },
      ]);
      const controller = new BrowserUseController(transport);
      const agent = new BrowserUseAgent({ controller, zer0qRunner: runner, maxSteps: 5 });

      let plannerCalls = 0;
      const planner = ({ step }: { readonly step: number }) => {
        plannerCalls += 1;
        if (step === 1) {
          return { action: 'extract' as const, format: 'text' as const, tokensUsed: 30 };
        }
        return { action: 'done' as const, result: { answer: 'Holo v11 ready' }, tokensUsed: 20 };
      };

      const run1 = await agent.run({
        task: 'Read Holo v11 status',
        startUrl: 'https://awesome-os.local/search',
        planner,
      });
      assert.equal(run1.status, 'COMPLETED');
      assert.equal(run1.cached, false);
      assert.equal(run1.tokensUsed, 50);
      assert.equal(plannerCalls, 2);

      const run2 = await agent.run({
        task: 'Read Holo v11 status',
        startUrl: 'https://awesome-os.local/search',
        planner,
      });
      assert.equal(run2.status, 'CACHE_HIT');
      assert.equal(run2.cached, true);
      assert.equal(run2.tokensUsed, 0);
      assert.equal(plannerCalls, 2);
      assert.equal(run2.receipts[0]?.action, 'cas_hit');
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('BrowserUseAgent enforces maxSteps and maxTokens budget guards cleanly', async () => {
    const transport = new MockBrowserTransport([
      {
        url: 'https://awesome-os.local/loop',
        title: 'Loop',
        dom: { tag: 'main', children: [{ tag: 'button', text: 'Next' }] },
        textContent: 'Looping page',
      },
    ]);
    const controller = new BrowserUseController(transport);
    const agent = new BrowserUseAgent({ controller, maxSteps: 2, maxTokens: 50 });

    const capped = await agent.run({
      task: 'Exceed step cap',
      startUrl: 'https://awesome-os.local/loop',
      cachePolicy: 'bypass',
      planner: () => ({ action: 'scroll', direction: 'down', amountPx: 50, tokensUsed: 10 }),
    });
    assert.equal(capped.status, 'MAX_STEPS_EXCEEDED');
    assert.equal(capped.stepsExecuted, 2);

    const overBudget = await agent.run({
      task: 'Exceed token cap',
      startUrl: 'https://awesome-os.local/loop',
      maxTokens: 50,
      cachePolicy: 'bypass',
      planner: () => ({ action: 'extract', tokensUsed: 80 }),
    });
    assert.equal(overBudget.status, 'BUDGET_EXCEEDED');
    assert.equal(overBudget.receipts.at(-1)?.action, 'guard_stop');
  });
});
