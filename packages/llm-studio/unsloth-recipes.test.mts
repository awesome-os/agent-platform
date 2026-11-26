import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  GGUF_EXPORT_QUANTS,
  LORA_TARGET_PRESETS,
  UNSLOTH_RECIPES,
  UnslothRecipeEngine,
  buildUnslothExportPlan,
  estimateGgufSizeMb,
  estimateTrainingVramGb,
  validateDataset,
} from './unsloth-recipes.mts';

describe('UnslothRecipeEngine — Recipes, Presets & Quantization Matrix', () => {
  const engine = new UnslothRecipeEngine();

  it('exposes all 7 canonical Unsloth fine-tuning recipes with expected invariants', () => {
    const ids = Object.keys(UNSLOTH_RECIPES);
    assert.deepEqual(ids.sort(), [
      'dpo',
      'embed-jina-v3-lora',
      'grpo',
      'kto',
      'orpo',
      'qlora-nf4',
      'sft-lora',
    ]);

    assert.equal(UNSLOTH_RECIPES['sft-lora'].sequencePacking, true);
    assert.equal(UNSLOTH_RECIPES['sft-lora'].useGradientCheckpointing, 'unsloth');

    assert.equal(UNSLOTH_RECIPES['qlora-nf4'].loadIn4Bit, true);
    assert.equal(UNSLOTH_RECIPES['qlora-nf4'].dynamicQuantization2_0, true);
    assert.ok(UNSLOTH_RECIPES['qlora-nf4'].preservedHighPrecisionLayers.includes('lm_head'));

    assert.equal(UNSLOTH_RECIPES.dpo.trainerClass, 'DPOTrainer');
    assert.equal(UNSLOTH_RECIPES.grpo.trainerClass, 'GRPOTrainer');
    assert.equal(UNSLOTH_RECIPES.orpo.trainerClass, 'ORPOTrainer');
    assert.equal(UNSLOTH_RECIPES.kto.trainerClass, 'KTOTrainer');

    const jina = UNSLOTH_RECIPES['embed-jina-v3-lora'];
    assert.equal(jina.defaultTargetPreset, 'jina-v3-task');
    assert.deepEqual(jina.hyperparameters.adapter_strategies, [
      'merge-then-stack',
      'direct-task-finetune',
    ]);
    assert.ok(Array.isArray(jina.hyperparameters.matryoshka_dims));

    assert.equal(engine.listRecipes('preference').length, 3);
    assert.equal(engine.listRecipes('embedding').length, 1);
  });

  it('defines exact LoRA target module presets', () => {
    assert.deepEqual(LORA_TARGET_PRESETS['attention-only'], [
      'q_proj',
      'k_proj',
      'v_proj',
      'o_proj',
    ]);
    assert.deepEqual(LORA_TARGET_PRESETS['all-linear'], [
      'q_proj',
      'k_proj',
      'v_proj',
      'o_proj',
      'gate_proj',
      'up_proj',
      'down_proj',
    ]);
    assert.deepEqual(LORA_TARGET_PRESETS['jina-v3-task'], ['wqkv', 'wo', 'gated_layers']);
    assert.deepEqual(engine.getTargetModules('jina-v3-task'), ['wqkv', 'wo', 'gated_layers']);
  });

  it('calculates monotonic GGUF output sizes across all 6 quantization tiers', () => {
    const ordered = ['Q4_K_M', 'UD-Q4_K_XL', 'Q5_K_M', 'Q6_K', 'Q8_0', 'F16'] as const;
    const sizes8B = ordered.map((q) => estimateGgufSizeMb(8, q));

    for (let i = 1; i < sizes8B.length; i++) {
      assert.ok(
        sizes8B[i] > sizes8B[i - 1],
        `Expected ${ordered[i]} (${sizes8B[i]} MB) > ${ordered[i - 1]} (${sizes8B[i - 1]} MB)`,
      );
    }

    assert.equal(GGUF_EXPORT_QUANTS['UD-Q4_K_XL'].isDynamic2_0, true);
    assert.equal(GGUF_EXPORT_QUANTS['UD-Q4_K_XL'].estimateSizeMb(8), estimateGgufSizeMb(8, 'UD-Q4_K_XL'));
    assert.ok(sizes8B[0] > 4500 && sizes8B[0] < 5200, `8B Q4_K_M should be ~4.7 GB, got ${sizes8B[0]} MB`);
    assert.ok(sizes8B[5] > 15000 && sizes8B[5] < 16000, `8B F16 should be ~15.5 GB, got ${sizes8B[5]} MB`);

    assert.throws(() => estimateGgufSizeMb(0, 'Q4_K_M'), RangeError);
  });
});

describe('validateDataset — All 5 Dataset Formats', () => {
  it('validates and normalizes alpaca rows', () => {
    const res = validateDataset('alpaca', [
      { instruction: 'Summarize the patch.', input: 'Removed unused import.', output: 'Cleaned up dead code.' },
      { instruction: 'Say hi.', output: 'Hello!' },
    ]);
    assert.equal(res.valid, true);
    assert.equal(res.rowCount, 2);
    assert.equal(res.validRowCount, 2);
    assert.equal(res.errors.length, 0);
    assert.ok(res.tokenEstimate > 10);
    assert.equal(res.normalizedRows[1].format, 'alpaca');
  });

  it('validates sharegpt and chatml conversation formats', () => {
    const sharegpt = validateDataset('sharegpt', [
      {
        conversations: [
          { from: 'human', value: 'Explain QLoRA.' },
          { from: 'gpt', value: 'QLoRA quantizes frozen base weights to 4-bit NF4 while training FP16 LoRA adapters.' },
        ],
      },
    ]);
    assert.equal(sharegpt.valid, true);
    assert.equal(sharegpt.normalizedRows.length, 1);

    const chatml = validateDataset('chatml', [
      {
        messages: [
          { role: 'system', content: 'You are a concise compiler assistant.' },
          { role: 'user', content: 'Fix TS2322.' },
          { role: 'assistant', content: 'Align the return type annotation.' },
        ],
      },
    ]);
    assert.equal(chatml.valid, true);
    assert.equal(chatml.normalizedRows.length, 1);

    const badChatml = validateDataset('chatml', [
      { messages: [{ role: 'invalid-role', content: 'oops' }] },
    ]);
    assert.equal(badChatml.valid, false);
    assert.equal(badChatml.errors[0].field, 'messages[0].role');
  });

  it('validates preference-dpo and contrastive-triplet datasets and rejects duplicates/empties', () => {
    const dpoOk = validateDataset('preference-dpo', [
      { prompt: 'Write a safe SQL query.', chosen: 'SELECT * FROM users WHERE id = $1;', rejected: 'SELECT * FROM users WHERE id = ' + '1;' },
    ]);
    assert.equal(dpoOk.valid, true);

    const dpoDup = validateDataset('preference-dpo', [
      { prompt: 'Hi', chosen: 'Same', rejected: 'Same' },
    ]);
    assert.equal(dpoDup.valid, false);
    assert.ok(dpoDup.errors.some((e) => e.field === 'rejected'));

    const tripletOk = validateDataset('contrastive-triplet', [
      {
        anchor: 'TypeError: Cannot read properties of undefined',
        positive: 'Add optional chaining before dereferencing nested property',
        negative: 'Increase HTTP server keep-alive timeout to 60s',
      },
    ]);
    assert.equal(tripletOk.valid, true);
    assert.equal(tripletOk.normalizedRows[0].format, 'contrastive-triplet');

    const emptyRes = validateDataset('alpaca', []);
    assert.equal(emptyRes.valid, false);
    assert.equal(emptyRes.rowCount, 0);
  });
});

describe('buildUnslothExportPlan — Sovereign Export & CloudDeviceMesh Dispatch', () => {
  const engine = new UnslothRecipeEngine();

  it('generates .b8p CAS manifest, Ollama Modelfile, llama.cpp CLI args, and GPU worker script for 8B GRPO', () => {
    const plan = buildUnslothExportPlan({
      modelId: 'unsloth/DeepSeek-R1-Distill-Llama-8B',
      adapterName: 'b8g-r1-reasoner',
      recipeId: 'grpo',
      paramBillions: 8,
      quant: 'UD-Q4_K_XL',
      systemPrompt: 'Think step by step inside <reasoning> tags.',
      contextLength: 4096,
      serverPort: 8091,
      cloudMeshNodeId: 'cuda-a100-node-1',
    });

    assert.equal(plan.b8pCasManifest.schema, 'b8p.adapter.v1');
    assert.match(plan.b8pCasManifest.digest, /^sha256:[0-9a-f]{64}$/);
    assert.equal(plan.b8pCasManifest.casPath, '.b8g-cas/adapters/b8g-r1-reasoner.b8p');
    assert.equal(plan.b8pCasManifest.dynamicQuantization2_0, true);

    assert.match(plan.ollamaModelfile, /^FROM \.\/exports\/b8g-r1-reasoner-UD-Q4_K_XL\.gguf/m);
    assert.match(plan.ollamaModelfile, /^ADAPTER \.\/exports\/b8g-r1-reasoner-lora\.gguf/m);
    assert.match(plan.ollamaModelfile, /^PARAMETER num_ctx 4096/m);
    assert.match(plan.ollamaModelfile, /SYSTEM """Think step by step inside <reasoning> tags\."""/);

    assert.equal(plan.llamaCpp.quantizeArgs[0], 'llama-quantize');
    assert.equal(plan.llamaCpp.quantizeArgs[3], 'UD-Q4_K_XL');
    assert.ok(plan.llamaCpp.serverArgs.includes('--lora'));
    assert.ok(plan.llamaCpp.serverArgs.includes('8091'));

    assert.ok(plan.gpuWorkerDispatch !== null);
    assert.equal(plan.gpuWorkerDispatch.targetMesh, 'CloudDeviceMesh');
    assert.equal(plan.gpuWorkerDispatch.nodeId, 'cuda-a100-node-1');
    assert.match(plan.gpuWorkerDispatch.pythonScript, /FastLanguageModel\.get_peft_model/);
    assert.match(plan.gpuWorkerDispatch.pythonScript, /use_gradient_checkpointing="unsloth"/);
    assert.match(plan.gpuWorkerDispatch.pythonScript, /GRPOTrainer/);
    assert.match(plan.gpuWorkerDispatch.pythonScript, /save_pretrained_gguf/);
  });

  it('supports Jina-v3 embedding LoRA with merge-then-stack, Matryoshka dims, --embedding flag, and skips GPU worker for sub-7B by default', () => {
    const plan = engine.buildExportPlan({
      modelId: 'jinaai/jina-embeddings-v3',
      adapterName: 'lumibert-error-repair',
      recipeId: 'embed-jina-v3-lora',
      paramBillions: 0.57,
      quant: 'Q8_0',
      embedAdapterStrategy: 'merge-then-stack',
      matryoshkaDims: [64, 32, 16],
    });

    assert.deepEqual(plan.b8pCasManifest.targetModules, ['wqkv', 'wo', 'gated_layers']);
    assert.deepEqual(plan.b8pCasManifest.embedding, {
      strategy: 'merge-then-stack',
      matryoshkaDims: [64, 32, 16],
    });
    assert.ok(plan.llamaCpp.serverArgs.includes('--embedding'));
    assert.equal(plan.gpuWorkerDispatch, null);
  });

  it('selects appropriate recipes and GPU tiers for 70B models', () => {
    const vram70bQlora = estimateTrainingVramGb(70, 'qlora-nf4');
    const vram70bSft = estimateTrainingVramGb(70, 'sft-lora');
    assert.ok(vram70bQlora < vram70bSft);

    const plan70b = engine.buildExportPlan({
      modelId: 'unsloth/Llama-3.3-70B-Instruct',
      adapterName: 'b8g-70b-qlora',
      recipeId: 'qlora-nf4',
      paramBillions: 70,
      quant: 'UD-Q4_K_XL',
    });
    assert.ok(plan70b.gpuWorkerDispatch !== null);
    assert.equal(plan70b.gpuWorkerDispatch.nodeTier, 'cuda-80gb');

    assert.equal(engine.recommendRecipe({ datasetFormat: 'contrastive-triplet' }).id, 'embed-jina-v3-lora');
    assert.equal(engine.recommendRecipe({ datasetFormat: 'alpaca', reasoningRl: true }).id, 'grpo');
    assert.equal(engine.recommendRecipe({ datasetFormat: 'preference-dpo', unpairedFeedback: true }).id, 'kto');
    assert.equal(engine.recommendRecipe({ datasetFormat: 'preference-dpo', availableVramGb: 12 }).id, 'orpo');
    assert.equal(engine.recommendRecipe({ datasetFormat: 'alpaca', paramBillions: 14, availableVramGb: 10 }).id, 'qlora-nf4');
  });
});
