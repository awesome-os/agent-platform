import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { describe, it } from 'node:test';
import {
  AgentEvalHarness,
  Crystallizer,
  FlightRecorder,
  SimulationRunner,
} from './index.mts';

describe('packages/observability-eval — FlightRecorder, EvalHarness, SimulationRunner & Crystallizer', () => {
  it('FlightRecorder logs all 6 event kinds, serializes NDJSON, and computes p50/p95 latency stats', async () => {
    const recorder = new FlightRecorder();
    const kinds = ['cas_hit', 'tool_call', 'moe_quarantine', 'armor_block', 'budget_tick', 'eval_score'] as const;
    const latencies = [0.4, 0.8, 1.2, 2.0, 3.5, 10.0];
    for (let i = 0; i < kinds.length; i += 1) {
      recorder.record({
        kind: kinds[i]!,
        runId: 'run-obs-1',
        latencyMs: latencies[i]!,
        tokensUsed: i === 1 ? 120 : 0,
        tokensSaved: i === 0 ? 500 : 0,
      });
    }

    const summary = recorder.getSummary();
    assert.equal(summary.totalEvents, 6);
    assert.equal(summary.totalTokensUsed, 120);
    assert.equal(summary.totalTokensSaved, 500);
    assert.ok(summary.overallLatency.p50Ms >= 0.8 && summary.overallLatency.p50Ms <= 2.0);
    assert.ok(summary.overallLatency.p95Ms >= 3.5);

    const ndjson = recorder.toNdjson();
    const parsed = FlightRecorder.fromNdjson(ndjson);
    assert.equal(parsed.length, 6);

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'obs-eval-'));
    try {
      const logPath = path.join(tmpDir, 'flight.ndjson');
      recorder.writeToFile(logPath);
      const fromFile = FlightRecorder.readFromFile(logPath);
      assert.equal(fromFile.length, 6);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }

    const stream = recorder.createEventStream();
    const reader = stream.getReader();
    let lineCount = 0;
    while (true) {
      const { done } = await reader.read();
      if (done) break;
      lineCount += 1;
    }
    assert.equal(lineCount, 6);
  });

  it('AgentEvalHarness and SimulationRunner run deterministic evaluations and capture invariants', async () => {
    const recorder = new FlightRecorder();
    const harness = new AgentEvalHarness(recorder);
    const report = await harness.run(
      'math-eval',
      [
        { id: 'c1', input: 2, expected: 4 },
        { id: 'c2', input: 5, expected: 25 },
      ],
      (n) => n * n,
    );
    assert.equal(report.totalCases, 2);
    assert.equal(report.passedCases, 2);
    assert.equal(report.passRate, 1.0);
    assert.equal(recorder.listEvents('eval_score').length, 2);

    const sim = new SimulationRunner(recorder);
    const simReport = await sim.runScenario(
      'cas-hit-flow',
      { tokensUsed: 0, casHits: 0, quarantined: false },
      [
        {
          name: 'skimmer-hit',
          eventKind: 'cas_hit',
          apply: (s) => ({ ...s, casHits: s.casHits + 1 }),
          invariant: (s) => s.tokensUsed === 0 && s.casHits === 1,
        },
        {
          name: 'quarantine-bad-expert',
          eventKind: 'moe_quarantine',
          apply: (s) => ({ ...s, quarantined: true }),
          invariant: (s) => s.quarantined === true,
        },
      ],
    );
    assert.equal(simReport.allInvariantsHeld, true);
    assert.equal(simReport.stepsExecuted, 2);
  });

  it('Crystallizer resolves exact and near-match distilled solutions in < 3 ms', () => {
    const crys = new Crystallizer<string>();
    crys.distill('Fix TS1294 erasableSyntaxOnly enum error in index.mts', 'Replace enum with const object + union type', {
      sourceModel: 'gemini-2.5-pro',
      confidence: 0.99,
    });

    const exact = crys.lookup('Fix TS1294 erasableSyntaxOnly enum error in index.mts');
    assert.equal(exact.hit, true);
    assert.equal(exact.similarity, 1.0);
    assert.ok(exact.latencyMs < 3.0);

    const near = crys.lookup('Fix TS1294 erasableSyntaxOnly enum error in index.ts', 0.7);
    assert.equal(near.hit, true);
    assert.ok(near.similarity >= 0.7);
    assert.ok(near.latencyMs < 3.0);

    const miss = crys.lookup('Completely unrelated weather forecast for Tokyo', 0.7);
    assert.equal(miss.hit, false);
    assert.equal(crys.getStats().hits, 2);
    assert.equal(crys.getStats().misses, 1);
  });
});
