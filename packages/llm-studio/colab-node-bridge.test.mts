import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { CloudDeviceMesh } from './cloud-devices.mts';
import {
  COLAB_GPU_TIER_SPECS,
  ColabNodeBridge,
  generateColabNotebookIpynb,
  parseUniversalNotebookCells,
  registerColabHostNode,
  type ColabGpuTier,
} from './colab-node-bridge.mts';

describe('registerColabHostNode & CloudDeviceMesh integration', () => {
  it('registers all Colab GPU/TPU tiers with Unified Device-IDs and 1024-slot shard positions', () => {
    const mesh = new CloudDeviceMesh({ autoRegisterLocalHost: false });
    const tiers: ColabGpuTier[] = ['T4-16GB', 'L4-24GB', 'A100-40GB', 'A100-80GB', 'TPU-v2-8'];

    for (const tier of tiers) {
      const device = registerColabHostNode(mesh, 'tenant-ai', {
        label: `colab-worker-${tier.toLowerCase()}`,
        gpuTier: tier,
        colabUrl: `https://colab.research.google.com/drive/test-${tier}`,
        tunnelEndpoint: `quic://colab-${tier.toLowerCase()}.b8g.internal:4433`,
      });

      assert.match(device.deviceId, /^b8g-dev-cloud-[0-9a-f]{16}$/);
      assert.equal(device.tenantId, 'tenant-ai');
      assert.equal(device.locality, 'public-cloud');
      assert.equal(device.transport, 'quic-tunnel');
      assert.equal(device.authMode, 'oauth-oidc');
      assert.equal(device.status, 'online');
      assert.ok(device.shardSlot >= 0 && device.shardSlot < 1024);
      assert.equal(device.hardware.gpuModel, COLAB_GPU_TIER_SPECS[tier].gpuModel);
      assert.equal(device.hardware.cpuCores, COLAB_GPU_TIER_SPECS[tier].cpuCores);
      assert.equal(device.hardware.ramMb, COLAB_GPU_TIER_SPECS[tier].ramMb);
      assert.equal(device.shardWeight, COLAB_GPU_TIER_SPECS[tier].shardWeight);
      assert.equal(device.endpoint, `quic://colab-${tier.toLowerCase()}.b8g.internal:4433`);
    }

    const listed = mesh.listDevices('tenant-ai');
    assert.equal(listed.length, tiers.length);

    const route = mesh.resolveShardOwner('lora-job:qwen-7b-sft', 'tenant-ai', 2);
    assert.ok(route.primary !== null);
    assert.match(route.primary.deviceId, /^b8g-dev-cloud-/);
    assert.equal(route.replicas.length, 1);
  });

  it('falls back to colabUrl or deterministic QUIC endpoint and validates empty label', () => {
    const mesh = new CloudDeviceMesh({ autoRegisterLocalHost: false });

    const withUrl = registerColabHostNode(mesh, 'tenant-1', {
      label: 'Research GPU',
      gpuTier: 'L4-24GB',
      colabUrl: 'https://colab.research.google.com/drive/abc123',
    });
    assert.equal(withUrl.endpoint, 'https://colab.research.google.com/drive/abc123');

    const withDefaultEndpoint = registerColabHostNode(mesh, 'tenant-1', {
      label: 'Batch Trainer',
      gpuTier: 'A100-80GB',
    });
    assert.equal(withDefaultEndpoint.endpoint, 'quic://colab-batch-trainer-a100-80gb.colab.b8g.internal:4433');

    assert.throws(
      () => registerColabHostNode(mesh, 'tenant-1', { label: '   ', gpuTier: 'T4-16GB' }),
      /must not be empty/i,
    );
  });
});

describe('generateColabNotebookIpynb (nbformat v4.5)', () => {
  it('emits valid nbformat v4.5 JSON with 5 cells for Unsloth QLoRA NF4 training', () => {
    const nb = generateColabNotebookIpynb({
      recipe: 'qlora-nf4',
      baseModel: 'unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit',
      rank: 32,
      quant: 'q4_k_m',
      datasetJson: JSON.stringify([{ prompt: 'hi', completion: 'hello' }]),
      callbackEndpoint: 'https://host.b8g.internal/api/colab/callback',
      deviceId: 'b8g-dev-cloud-1122334455667788',
      gpuTier: 'A100-40GB',
    });

    assert.equal(nb.nbformat, 4);
    assert.equal(nb.nbformat_minor, 5);
    assert.equal(nb.metadata.kernelspec.name, 'python3');
    assert.equal(nb.metadata.accelerator, 'GPU');
    assert.equal(nb.metadata.colab.gpuType, 'A100');
    assert.equal(nb.metadata.awesome_os.deviceId, 'b8g-dev-cloud-1122334455667788');
    assert.equal(nb.cells.length, 5);

    // Every nbformat 4.5 cell must carry a valid non-empty id
    for (const cell of nb.cells) {
      assert.match(cell.id, /^[a-zA-Z0-9-_]+$/);
    }

    assert.equal(nb.cells[0]?.cell_type, 'markdown');
    const headerSrc = nb.cells[0]!.source.join('');
    assert.match(headerSrc, /Awesome-OS Sovereign Colab Host Node Intersection/);
    assert.match(headerSrc, /b8g-dev-cloud-1122334455667788/);

    const installSrc = nb.cells[1]!.source.join('');
    assert.match(installSrc, /unsloth/);

    const loadSrc = nb.cells[2]!.source.join('');
    assert.match(loadSrc, /FastLanguageModel\.from_pretrained/);
    assert.match(loadSrc, /LORA_RANK = 32/);

    const trainSrc = nb.cells[3]!.source.join('');
    assert.match(trainSrc, /SFTTrainer/);
    assert.match(trainSrc, /paged_adamw_8bit/);

    const exportSrc = nb.cells[4]!.source.join('');
    assert.match(exportSrc, /save_pretrained_gguf/);
    assert.match(exportSrc, /b8g-adapter\.b8p/);
    assert.match(exportSrc, /hashlib\.sha256/);
    assert.match(exportSrc, /\.b8g-cas/);
    assert.match(exportSrc, /https:\/\/host\.b8g\.internal\/api\/colab\/callback/);
  });

  it('supports DPO, GRPO, and Jina-v3 Task-LoRA merge-then-stack & direct-task-finetune recipes', () => {
    const dpoNb = generateColabNotebookIpynb({
      recipe: 'dpo',
      baseModel: 'unsloth/Llama-3.2-3B-Instruct',
    });
    assert.match(dpoNb.cells[3]!.source.join(''), /DPOTrainer/);

    const grpoNb = generateColabNotebookIpynb({
      recipe: 'grpo',
      baseModel: 'unsloth/Qwen2.5-3B-Instruct',
    });
    assert.match(grpoNb.cells[3]!.source.join(''), /GRPOTrainer/);

    const jinaMergeNb = generateColabNotebookIpynb({
      recipe: 'jina-v3-merge-then-stack',
      baseModel: 'jinaai/jina-embeddings-v3',
      taskAdapter: 'retrieval.passage',
      rank: 8,
      quant: 'q8_0',
    });
    assert.match(jinaMergeNb.cells[2]!.source.join(''), /AutoModel\.from_pretrained/);
    assert.match(jinaMergeNb.cells[2]!.source.join(''), /retrieval\.passage/);
    assert.match(jinaMergeNb.cells[3]!.source.join(''), /merge_and_unload/);

    const jinaDirectNb = generateColabNotebookIpynb({
      recipe: 'jina-v3-direct-task-finetune',
      baseModel: 'jinaai/jina-embeddings-v3',
      taskAdapter: 'classification',
    });
    assert.match(jinaDirectNb.cells[3]!.source.join(''), /direct-task-finetune/);
    assert.match(jinaDirectNb.cells[4]!.source.join(''), /b8g-adapter\.b8p/);
  });
});

describe('parseUniversalNotebookCells (.ipynb & executable Markdown)', () => {
  it('parses a generated .ipynb JSON document into uniform NotebookCell[]', () => {
    const nb = generateColabNotebookIpynb({
      recipe: 'sft',
      baseModel: 'unsloth/Qwen2.5-7B-Instruct',
      deviceId: 'b8g-dev-cloud-aabbccddeeff0011',
    });

    const cells = parseUniversalNotebookCells(JSON.stringify(nb));
    assert.equal(cells.length, 5);
    assert.equal(cells[0]?.kind, 'markdown');
    assert.equal(cells[0]?.executable, false);
    assert.equal(cells[0]?.language, 'markdown');
    assert.equal(cells[0]?.id, 'b8g-cell-01-header');

    for (let i = 1; i < 5; i++) {
      assert.equal(cells[i]?.kind, 'code');
      assert.equal(cells[i]?.executable, true);
      assert.equal(cells[i]?.language, 'python');
      assert.equal(cells[i]?.metadata.sourceFormat, 'ipynb');
    }
  });

  it('parses executable Markdown fenced code cells (python, ts, bash) alongside prose', () => {
    const md = [
      '# Universal Notebook Intersection',
      '',
      'Run local preflight checks before dispatching to Colab.',
      '',
      '```bash id=preflight',
      'echo "checking .b8g-cas"',
      '```',
      '',
      'Next, prepare the training dataset in TypeScript:',
      '',
      '```ts id=prep-data title="Dataset Prep"',
      'const rows = [{ prompt: "ping", completion: "pong" }];',
      'console.log(JSON.stringify(rows));',
      '```',
      '',
      '```python',
      'import torch',
      'print(torch.cuda.is_available())',
      '```',
      '',
      '```json',
      '{"note": "non-executable config block"}',
      '```',
    ].join('\n');

    const cells = parseUniversalNotebookCells(md);
    assert.equal(cells.length, 6);

    assert.equal(cells[0]?.kind, 'markdown');
    assert.equal(cells[0]?.executable, false);
    assert.match(cells[0]!.source, /Universal Notebook Intersection/);

    assert.equal(cells[1]?.kind, 'code');
    assert.equal(cells[1]?.id, 'preflight');
    assert.equal(cells[1]?.language, 'bash');
    assert.equal(cells[1]?.executable, true);

    assert.equal(cells[2]?.kind, 'markdown');
    assert.equal(cells[2]?.executable, false);

    assert.equal(cells[3]?.kind, 'code');
    assert.equal(cells[3]?.id, 'prep-data');
    assert.equal(cells[3]?.language, 'ts');
    assert.equal(cells[3]?.executable, true);
    assert.equal(cells[3]?.metadata.title, 'Dataset Prep');

    assert.equal(cells[4]?.kind, 'code');
    assert.equal(cells[4]?.language, 'python');
    assert.equal(cells[4]?.executable, true);

    assert.equal(cells[5]?.kind, 'code');
    assert.equal(cells[5]?.language, 'json');
    assert.equal(cells[5]?.executable, false);
  });

  it('handles empty input cleanly', () => {
    assert.deepEqual(parseUniversalNotebookCells(''), []);
    assert.deepEqual(parseUniversalNotebookCells('   \n\t  '), []);
  });
});

describe('ColabNodeBridge coordinator class', () => {
  it('registers nodes, routes jobs on the 1024-slot ring, and builds/parses notebooks end-to-end', () => {
    const bridge = new ColabNodeBridge({ defaultTenantId: 'org-sovereign' });
    const node = bridge.registerNode({
      label: 'Primary A100 Trainer',
      gpuTier: 'A100-80GB',
      tunnelEndpoint: 'quic://a100-primary.colab.b8g.internal:4433',
    });

    const nodes = bridge.listColabNodes('org-sovereign');
    assert.equal(nodes.length, 1);
    assert.equal(nodes[0]?.device.deviceId, node.deviceId);
    assert.equal(nodes[0]?.colab.gpuTier, 'A100-80GB');

    const route = bridge.routeJobToColab('job:jina-v3-finetune', 'org-sovereign');
    assert.equal(route.primary?.deviceId, node.deviceId);

    const doc = bridge.buildNotebook({
      recipe: 'sft',
      baseModel: 'unsloth/Qwen2.5-Coder-7B-Instruct',
      deviceId: node.deviceId,
    });
    assert.equal(doc.metadata.colab.gpuType, 'A100');
    assert.equal(doc.metadata.awesome_os.callbackEndpoint, 'quic://a100-primary.colab.b8g.internal:4433');

    const serialized = bridge.serializeNotebook(doc);
    const roundTripped = bridge.parseNotebook(serialized);
    assert.equal(roundTripped.length, 5);
    assert.equal(roundTripped[0]?.kind, 'markdown');
    assert.equal(roundTripped[4]?.kind, 'code');
  });
});
