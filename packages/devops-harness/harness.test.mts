/**
 * @file intersections/b8g-devops-harness/harness.test.mts
 * @description Unit tests for `intersections/b8g-devops-harness` (1:1 Error-to-Solution
 * routing, deterministic verifier gate, `b8g-devops` P2 escalation, and `crystallizeFix()`).
 */

import assert from 'node:assert/strict';
import { beforeEach, describe, it } from 'node:test';
import {
  B8G_DEVOPS_CONTEXT,
  clearDevOpsHub,
  getDevOpsSnapshot,
  registerEmergencyWorker,
  type DevOpsEvent,
} from './devops.mts';
import {
  B8gDevOpsHarnessIntersection,
  DevOpsErrorHarness,
  ErrorFixCasSection,
  SEEDED_ERROR_FIXES,
  type ErrorFixRecord,
} from './index.mts';

describe('intersections/b8g-devops-harness', () => {
  beforeEach(() => {
    clearDevOpsHub();
  });

  it('composes B8gDevOpsHarnessIntersection cleanly with LumiBertSection, ErrorFixCasSection, and B8gDevOpsIntersection', () => {
    assert.equal(ErrorFixCasSection.id, 'b8g/error-fix-cas');
    assert.equal(B8gDevOpsHarnessIntersection.id, 'b8g/devops-harness');
    assert.equal(SEEDED_ERROR_FIXES.length, 8);
  });

  it('maps noisy real-world error variants 1:1 to the exact seeded solution', () => {
    const harness = new DevOpsErrorHarness();

    const cases: readonly { readonly noisy: string; readonly expectedId: string }[] = [
      {
        noisy: 'Error: listen EADDRINUSE: address already in use 127.0.0.1:8890',
        expectedId: 'fix-eaddrinuse',
      },
      {
        noisy: 'vs_toolchain.py failed: set DEPOT_TOOLS_WIN_TOOLCHAIN=0 to use local Visual Studio',
        expectedId: 'fix-depot-tools-win-toolchain',
      },
      {
        noisy: 'ERROR at //.gn:18:21: This is not a string. script_executable = ["python3.exe"]',
        expectedId: 'fix-gn-script-executable',
      },
      {
        noisy: 'stealify-lint B8G016: forbidden http.createServer call in local IPC handler',
        expectedId: 'fix-b8g016-http-create-server',
      },
      {
        noisy: 'b8g-web-doc [B8G018] HTML table column mismatch: <tr> has 3 <td> cells vs 4 <th> header cells',
        expectedId: 'fix-b8g018-html-table-columns',
      },
      {
        noisy: 'b8g-web-doc [B8G019] Markdown table unescaped pipe character caused column mismatch',
        expectedId: 'fix-b8g019-markdown-pipe',
      },
      {
        noisy: 'error TS1294: This syntax is not allowed when erasableSyntaxOnly is enabled (enum Status)',
        expectedId: 'fix-erasable-syntax-only-enum',
      },
      {
        noisy: 'gn gen failed: third_party/rust/read_fonts DEPS drift Cargo.toml BUILD.gn mismatch',
        expectedId: 'fix-chromium-read-fonts-deps',
      },
    ];

    for (const testCase of cases) {
      const res = harness.resolveError1To1(testCase.noisy);
      assert.equal(
        res.status,
        'AUTO_FIXED',
        `expected AUTO_FIXED for "${testCase.noisy}", got ${JSON.stringify(res)}`,
      );
      assert.equal(res.fix.id, testCase.expectedId);
      assert.ok(res.latencyMs < 20, `unexpectedly slow resolve: ${res.latencyMs}ms`);
    }
  });

  it('escalates to LLM and emits P2/dev-error-loop when deterministic verifyFix gate rejects', () => {
    const harness = new DevOpsErrorHarness();
    const received: DevOpsEvent[] = [];
    const dispose = registerEmergencyWorker('dev-error-loop', (ev) => {
      received.push(ev);
    });

    try {
      const res = harness.resolveError1To1(
        'Error: listen EADDRINUSE: address already in use 127.0.0.1:8890',
        {
          verifyFix: () => false,
          callerCtx: B8G_DEVOPS_CONTEXT,
        },
      );

      assert.equal(res.status, 'ESCALATE_TO_LLM');
      assert.equal(res.bestCandidateId, 'fix-eaddrinuse');
      assert.equal(received.length, 1);
      assert.equal(received[0]!.severity, 'P2');
      assert.equal(received[0]!.workerTarget, 'dev-error-loop');
      assert.equal(received[0]!.code, 'HARNESS_VERIFIER_REJECTED');

      const snap = getDevOpsSnapshot();
      assert.equal(snap.totalEmitted, 1);
    } finally {
      dispose();
    }
  });

  it('escalates unknown novel errors cleanly and resolves them immediately after crystallizeFix()', () => {
    const harness = new DevOpsErrorHarness();
    const novelError =
      'VulkanSwapchainOutOfDateKHR: vkQueuePresentKHR returned -1000001004 during Wayland resize';

    const firstAttempt = harness.resolveError1To1(novelError, {
      callerCtx: B8G_DEVOPS_CONTEXT,
    });
    assert.equal(firstAttempt.status, 'ESCALATE_TO_LLM');

    const snap = getDevOpsSnapshot();
    assert.equal(snap.totalEmitted, 1);
    assert.equal(snap.events[0]!.severity, 'P2');
    assert.equal(snap.events[0]!.workerTarget, 'dev-error-loop');

    const newFix: ErrorFixRecord = {
      id: 'fix-vulkan-swapchain-out-of-date',
      code: 'VK_ERROR_OUT_OF_DATE_KHR',
      category: 'gpu-vulkan',
      canonicalError: novelError,
      summary: 'Recreate Vulkan swapchain and framebuffers on Wayland surface resize.',
      resolutionSteps: [
        'Call vkDeviceWaitIdle(device).',
        'Destroy stale swapchain image views and recreate swapchain with updated wl_surface extent.',
      ],
    };

    harness.crystallizeFix(novelError, newFix);

    // Exact and noisy variant both auto-fix in < 5 ms with 0 LLM tokens
    const exactRes = harness.resolveError1To1(novelError);
    assert.equal(exactRes.status, 'AUTO_FIXED');
    assert.equal(exactRes.fix.id, 'fix-vulkan-swapchain-out-of-date');
    assert.ok(exactRes.cosine > 0.99);

    const noisyVariantRes = harness.resolveError1To1(
      'VulkanSwapchainOutOfDateKHR vkQueuePresentKHR failed on Wayland resize',
    );
    assert.equal(noisyVariantRes.status, 'AUTO_FIXED');
    assert.equal(noisyVariantRes.fix.id, 'fix-vulkan-swapchain-out-of-date');
    assert.ok(noisyVariantRes.latencyMs < 5.0, `expected < 5ms, got ${noisyVariantRes.latencyMs}ms`);
  });

  it('benchmark reports 100% auto-fix rate and < 5ms average latency', () => {
    const harness = new DevOpsErrorHarness();
    const report = harness.benchmark(24);
    assert.equal(report.iterations, 24);
    assert.equal(report.autoFixedRate, 1.0);
    assert.equal(report.isPretrainedWeightsLoaded, false);
    assert.ok(report.avgLatencyMs < 5.0, `expected < 5ms avg, got ${report.avgLatencyMs.toFixed(3)}ms`);
  });
});
