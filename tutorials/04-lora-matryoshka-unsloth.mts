/**
 * @file tutorials/04-lora-matryoshka-unsloth.mts
 * @description Tutorial 04: Run real InfoNCE contrastive LoRA training, slice Matryoshka
 * embeddings, merge `W' = W + (alpha/r) B A`, generate an Unsloth GGUF + `.b8p` CAS export
 * plan, and dispatch an `.ipynb` notebook through `ColabNodeBridge`.
 */

import assert from 'node:assert/strict';
import {
  LoraStudioEngine,
  cosineSimilarity,
  estimateUnslothMemory,
  sliceMatryoshka,
  type ContrastivePair,
} from '../packages/llm-studio/lora-engine.mts';
import {
  UnslothRecipeEngine,
  buildUnslothExportPlan,
} from '../packages/llm-studio/unsloth-recipes.mts';
import { ColabNodeBridge } from '../packages/llm-studio/colab-node-bridge.mts';

async function main(): Promise<void> {
  console.log('=== Tutorial 04: InfoNCE LoRA, Matryoshka, Unsloth & Colab GPU Mesh ===');

  // 1. Contrastive LoRA + Matryoshka Slicing
  const studio = new LoraStudioEngine({ dim: 64, defaultRank: 8 });
  const pairs: ContrastivePair[] = [
    {
      anchor: 'boundary verified moe gate capability token',
      positive: 'section isolated expert router with ingress egress norm guard',
      negative: 'legacy uncached docker image tarball download',
    },
    {
      anchor: 'atlaskv hierarchical key value pruning arena',
      positive: 'sublinear cluster centroid lookup over flat float32 array',
      negative: 'css flexbox responsive grid breakpoint tutorial',
    },
  ];

  const stream = studio.trainStream('retrieval.query', pairs, { steps: 6, lr: 0.12 });
  const reader = stream.getReader();
  const losses: number[] = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    losses.push(value.loss);
  }
  console.log(`LoRA 6-step losses: ${losses.map((l) => l.toFixed(4)).join(' -> ')}`);
  assert.ok(losses[losses.length - 1]! <= losses[0]!, 'Expected contrastive training loss to decrease or hold');

  const fullVec = studio.embed('boundary verified moe gate', { taskId: 'retrieval.query' });
  const mat32 = sliceMatryoshka(fullVec, 32);
  const mat16 = sliceMatryoshka(fullVec, 16);
  const mat32Other = studio.embed('section isolated expert router', {
    taskId: 'retrieval.query',
    matryoshkaDim: 32,
  });
  console.log(`Matryoshka dims: full=${fullVec.length}, m32=${mat32.length}, m16=${mat16.length}, cos@32=${cosineSimilarity(mat32, mat32Other).toFixed(4)}`);

  const mergeRes = studio.mergeAdapter('retrieval.query');
  console.log(`Merged W' = W + (alpha/r)BA: ${mergeRes.bytes} B, sha256=${mergeRes.sha256.slice(0, 16)}...`);

  const vram = estimateUnslothMemory(8, 16, 'nf4', 4);
  console.log(`Unsloth NF4 VRAM (8B, r=16, bs=4): ${vram.totalMb} MB (${vram.savingsPctVsFp16}% saved vs FP16)`);

  // 2. Unsloth Recipe Engine & GGUF Export Plan
  const recipeEngine = new UnslothRecipeEngine();
  const recommended = recipeEngine.recommendRecipe({
    datasetFormat: 'chatml',
    availableVramGb: 16,
    paramBillions: 8,
  });
  const exportPlan = buildUnslothExportPlan({
    modelId: 'unsloth/Qwen3-8B-unsloth-bnb-4bit',
    paramBillions: 8,
    adapterName: 'b8g-moe-coder',
    recipeId: recommended.id,
    quant: 'UD-Q4_K_XL',
  });
  console.log(`Recommended Unsloth Recipe: ${recommended.id} -> CAS ${exportPlan.b8pCasManifest.casPath} (${exportPlan.estimatedOutputSizeMb} MB)`);

  // 3. Colab GPU Mesh & nbformat v4.5 Notebook Generation
  const bridge = new ColabNodeBridge();
  const node = bridge.registerNode({ label: 'colab-t4-01', gpuTier: 'T4-16GB' });
  const route = bridge.routeJobToColab('b8g-moe-coder');
  const nb = bridge.buildNotebook({
    recipe: 'qlora-nf4',
    baseModel: 'unsloth/Qwen3-8B-unsloth-bnb-4bit',
    deviceId: node.deviceId,
    rank: 16,
  });
  assert.equal(nb.nbformat, 4);
  assert.equal(nb.cells.length, 5);
  console.log(`Colab Node Registered: ${node.deviceId} -> Shard Primary: ${route.primary.deviceId} (${nb.cells.length} .ipynb cells)`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
