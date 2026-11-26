import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { detectHardware, recommendProfilesAndModels, type FitRecommendation } from './hardware-fit.mts';

describe('packages/hardware-fit — Hardware Detection & Sovereign Model Recommender', () => {
  it('detects host CPU/RAM/disk metrics and recommends fitting local LLM/MoE models', () => {
    const hw = detectHardware();
    assert.ok(hw.cpuCores >= 1);
    assert.ok(hw.memoryTotalGb > 0);
    assert.ok(hw.memoryFreeGb > 0);
    assert.ok(hw.hostname.length > 0);

    const recs: FitRecommendation[] = recommendProfilesAndModels(hw);
    const osProfiles = recs.filter((r) => r.kind === 'os-profile');
    const llmEngines = recs.filter((r) => r.kind === 'llm-engine');

    assert.ok(osProfiles.length >= 1);
    assert.ok(llmEngines.length >= 1);
    for (const item of recs) {
      assert.ok(item.id.length > 0);
      assert.ok(item.displayName.length > 0);
      assert.ok(item.ramMinGb <= item.ramGoodGb);
    }
  });
});
