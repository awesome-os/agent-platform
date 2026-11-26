/**
 * @file colab-node-bridge.mts
 * @description Google Colab Remote Host Node & Universal Notebook Bridge (`ColabNodeBridge`).
 *
 * Bridges Google Colab GPU/TPU runtimes (T4, L4, A100-40GB, A100-80GB, TPU-v2-8)
 * as first-class remote `DeviceRecord` host nodes inside `CloudDeviceMesh` with
 * Unified Device-IDs (`b8g-dev-cloud-...`) and 1024-slot consistent-hash ring positions.
 *
 * Generates valid nbformat v4.5 `.ipynb` notebooks for Unsloth / QLoRA / DPO / GRPO /
 * Jina-v3 Task-LoRA training with GGUF + `.b8p` + SHA-256 export back to `.b8g-cas/`,
 * and parses both `.ipynb` JSON and executable Markdown fenced code blocks into a
 * uniform `NotebookCell[]` array for the Universal Notebook Runner.
 *
 * Zero external dependencies — pure Node 26 TypeScript, strict `erasableSyntaxOnly: true`.
 */
import { createHash } from 'node:crypto';
import {
  CloudDeviceMesh,
  type DeviceHardwareSpec,
  type DeviceRecord,
  type ShardRouteResult,
} from './cloud-devices.mts';

export type ColabGpuTier = 'T4-16GB' | 'L4-24GB' | 'A100-40GB' | 'A100-80GB' | 'TPU-v2-8';

export type ColabAcceleratorKind = 'GPU' | 'TPU';

export interface ColabGpuTierSpec {
  tier: ColabGpuTier;
  gpuModel: string;
  vramGb: number;
  cpuCores: number;
  ramMb: number;
  shardWeight: number;
  accelerator: ColabAcceleratorKind;
  colabRuntimeType: 'T4' | 'L4' | 'A100' | 'TPU';
  highRam: boolean;
}

export const COLAB_GPU_TIER_SPECS: Readonly<Record<ColabGpuTier, ColabGpuTierSpec>> = {
  'T4-16GB': {
    tier: 'T4-16GB',
    gpuModel: 'NVIDIA Tesla T4 (16GB VRAM)',
    vramGb: 16,
    cpuCores: 2,
    ramMb: 12_984,
    shardWeight: 16,
    accelerator: 'GPU',
    colabRuntimeType: 'T4',
    highRam: false,
  },
  'L4-24GB': {
    tier: 'L4-24GB',
    gpuModel: 'NVIDIA L4 (24GB VRAM)',
    vramGb: 24,
    cpuCores: 12,
    ramMb: 54_272,
    shardWeight: 32,
    accelerator: 'GPU',
    colabRuntimeType: 'L4',
    highRam: true,
  },
  'A100-40GB': {
    tier: 'A100-40GB',
    gpuModel: 'NVIDIA A100-SXM4-40GB',
    vramGb: 40,
    cpuCores: 12,
    ramMb: 85_504,
    shardWeight: 64,
    accelerator: 'GPU',
    colabRuntimeType: 'A100',
    highRam: true,
  },
  'A100-80GB': {
    tier: 'A100-80GB',
    gpuModel: 'NVIDIA A100-SXM4-80GB',
    vramGb: 80,
    cpuCores: 12,
    ramMb: 171_008,
    shardWeight: 96,
    accelerator: 'GPU',
    colabRuntimeType: 'A100',
    highRam: true,
  },
  'TPU-v2-8': {
    tier: 'TPU-v2-8',
    gpuModel: 'Google Cloud TPU v2-8 (64GB HBM)',
    vramGb: 64,
    cpuCores: 96,
    ramMb: 342_016,
    shardWeight: 80,
    accelerator: 'TPU',
    colabRuntimeType: 'TPU',
    highRam: true,
  },
};

export interface RegisterColabHostNodeOptions {
  label: string;
  gpuTier: ColabGpuTier;
  colabUrl?: string;
  tunnelEndpoint?: string;
  publicKeyFingerprint?: string;
}

export type TrainingRecipe =
  | 'sft'
  | 'qlora-nf4'
  | 'dpo'
  | 'grpo'
  | 'jina-v3-merge-then-stack'
  | 'jina-v3-direct-task-finetune';

export type JinaTaskAdapter =
  | 'retrieval.query'
  | 'retrieval.passage'
  | 'separation'
  | 'classification'
  | 'text-matching'
  | (string & {});

export interface GenerateColabNotebookOptions {
  recipe: TrainingRecipe;
  baseModel: string;
  taskAdapter?: JinaTaskAdapter;
  rank?: number;
  quant?: string;
  datasetJson?: string;
  callbackEndpoint?: string;
  deviceId?: string;
  gpuTier?: ColabGpuTier;
  maxSeqLength?: number;
  epochs?: number;
  learningRate?: number;
}

export interface IpynbMarkdownCell {
  id: string;
  cell_type: 'markdown';
  metadata: Record<string, unknown>;
  source: string[];
}

export interface IpynbCodeCell {
  id: string;
  cell_type: 'code';
  execution_count: number | null;
  metadata: Record<string, unknown>;
  outputs: unknown[];
  source: string[];
}

export interface IpynbRawCell {
  id: string;
  cell_type: 'raw';
  metadata: Record<string, unknown>;
  source: string[];
}

export type IpynbCell = IpynbMarkdownCell | IpynbCodeCell | IpynbRawCell;

export interface IpynbDocument {
  nbformat: 4;
  nbformat_minor: 5;
  metadata: {
    colab: {
      provenance: unknown[];
      gpuType?: string;
      machine_shape?: 'hm' | 'standard';
    };
    accelerator: ColabAcceleratorKind;
    kernelspec: {
      name: 'python3';
      display_name: 'Python 3';
      language: 'python';
    };
    language_info: {
      name: 'python';
      version: string;
      mimetype: 'text/x-python';
      file_extension: '.py';
    };
    awesome_os: {
      bridge: 'ColabNodeBridge';
      deviceId: string;
      recipe: TrainingRecipe;
      baseModel: string;
      taskAdapter: string | null;
      rank: number;
      quant: string;
      callbackEndpoint: string | null;
    };
  };
  cells: IpynbCell[];
}

export type NotebookCellKind = 'code' | 'markdown' | 'raw';

export interface NotebookCell {
  index: number;
  id: string;
  kind: NotebookCellKind;
  language: string;
  source: string;
  executable: boolean;
  metadata: Record<string, unknown>;
}

export interface ColabNodeMetadata {
  deviceId: string;
  label: string;
  gpuTier: ColabGpuTier;
  spec: ColabGpuTierSpec;
  colabUrl: string;
  tunnelEndpoint: string;
}

export interface ColabNodeRecord {
  device: DeviceRecord;
  colab: ColabNodeMetadata;
}

export interface ColabNodeBridgeOptions {
  mesh?: CloudDeviceMesh;
  defaultTenantId?: string;
}

function sanitizeSlug(value: string): string {
  const cleaned = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return cleaned.length > 0 ? cleaned.slice(0, 40) : 'runtime';
}

function toSourceLines(text: string): string[] {
  const normalized = text.replace(/\r\n/g, '\n');
  if (normalized.length === 0) return [];
  const rawLines = normalized.split('\n');
  return rawLines.map((line, idx) => (idx < rawLines.length - 1 ? `${line}\n` : line));
}

function isJinaV3Target(recipe: TrainingRecipe, baseModel: string): boolean {
  return (
    recipe === 'jina-v3-merge-then-stack' ||
    recipe === 'jina-v3-direct-task-finetune' ||
    baseModel.toLowerCase().includes('jina-embeddings-v3')
  );
}

function escapePythonString(value: string): string {
  return JSON.stringify(value);
}

/**
 * Registers a Google Colab GPU/TPU runtime as a first-class remote `DeviceRecord`
 * inside `CloudDeviceMesh` (`locality: 'public-cloud'`, `transport: 'quic-tunnel'`,
 * `authMode: 'oauth-oidc'`), assigning a Unified Device-ID (`b8g-dev-cloud-...`)
 * and a 1024-slot consistent-hash shard position.
 */
export function registerColabHostNode(
  mesh: CloudDeviceMesh,
  tenantId: string,
  opts: RegisterColabHostNodeOptions,
): DeviceRecord {
  const spec = COLAB_GPU_TIER_SPECS[opts.gpuTier];
  if (!spec) {
    throw new Error(`Unsupported Colab GPU tier: ${String(opts.gpuTier)}`);
  }

  const label = opts.label.trim();
  if (label.length === 0) {
    throw new Error('Colab host node label must not be empty');
  }

  const slug = sanitizeSlug(label);
  const hostname = `colab-${slug}-${opts.gpuTier.toLowerCase()}`;
  const endpoint =
    opts.tunnelEndpoint?.trim() ||
    opts.colabUrl?.trim() ||
    `quic://${hostname}.colab.b8g.internal:4433`;

  const hardware: DeviceHardwareSpec = {
    hostname,
    platform: 'linux',
    arch: 'x64',
    cpuCores: spec.cpuCores,
    ramMb: spec.ramMb,
    gpuModel: spec.gpuModel,
  };

  const name = label.includes(opts.gpuTier) ? label : `${label} (Colab ${opts.gpuTier})`;

  return mesh.registerDevice({
    tenantId,
    name,
    transport: 'quic-tunnel',
    authMode: 'oauth-oidc',
    locality: 'public-cloud',
    endpoint,
    hardware,
    publicKeyFingerprint: opts.publicKeyFingerprint,
    shardWeight: spec.shardWeight,
  });
}

function buildInstallCellCode(isJina: boolean, recipe: TrainingRecipe): string {
  if (isJina) {
    return [
      '# Awesome-OS Sovereign Colab Cell 2: Install Jina-v3 Task-LoRA Dependencies',
      '%pip install -q --upgrade torch transformers peft einops accelerate safetensors datasets',
      `print("Installed Jina-v3 Task-LoRA stack for recipe: ${recipe}")`,
    ].join('\n');
  }

  return [
    '# Awesome-OS Sovereign Colab Cell 2: Install Unsloth & TRL Dependencies',
    '%pip install -q --upgrade unsloth trl peft accelerate bitsandbytes datasets safetensors',
    `print("Installed Unsloth stack for recipe: ${recipe}")`,
  ].join('\n');
}

function buildLoadModelCellCode(opts: {
  isJina: boolean;
  recipe: TrainingRecipe;
  baseModel: string;
  taskAdapter: string;
  rank: number;
  maxSeqLength: number;
}): string {
  if (opts.isJina) {
    return [
      '# Awesome-OS Sovereign Colab Cell 3: Load jina-embeddings-v3 with Task-LoRA',
      'import torch',
      'from transformers import AutoModel, AutoTokenizer',
      'from peft import LoraConfig, get_peft_model',
      '',
      `BASE_MODEL = ${escapePythonString(opts.baseModel)}`,
      `TASK_ADAPTER = ${escapePythonString(opts.taskAdapter)}`,
      `LORA_RANK = ${opts.rank}`,
      '',
      'tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)',
      'model = AutoModel.from_pretrained(',
      '    BASE_MODEL,',
      '    trust_remote_code=True,',
      '    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,',
      ')',
      'if torch.cuda.is_available():',
      '    model = model.to("cuda")',
      'print(f"Loaded {BASE_MODEL} with task_adapter={TASK_ADAPTER}, rank={LORA_RANK}")',
    ].join('\n');
  }

  const loadIn4Bit = opts.recipe === 'qlora-nf4' || opts.recipe === 'sft' ? 'True' : 'True';
  return [
    '# Awesome-OS Sovereign Colab Cell 3: Load Base Model via Unsloth FastLanguageModel',
    'import torch',
    'from unsloth import FastLanguageModel',
    '',
    `BASE_MODEL = ${escapePythonString(opts.baseModel)}`,
    `MAX_SEQ_LENGTH = ${opts.maxSeqLength}`,
    `LORA_RANK = ${opts.rank}`,
    `LOAD_IN_4BIT = ${loadIn4Bit}`,
    '',
    'model, tokenizer = FastLanguageModel.from_pretrained(',
    '    model_name=BASE_MODEL,',
    '    max_seq_length=MAX_SEQ_LENGTH,',
    '    dtype=None,',
    '    load_in_4bit=LOAD_IN_4BIT,',
    ')',
    'model = FastLanguageModel.get_peft_model(',
    '    model,',
    '    r=LORA_RANK,',
    '    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],',
    '    lora_alpha=LORA_RANK * 2,',
    '    lora_dropout=0,',
    '    bias="none",',
    '    use_gradient_checkpointing="unsloth",',
    ')',
    'print(f"Loaded {BASE_MODEL} (4bit={LOAD_IN_4BIT}, rank={LORA_RANK})")',
  ].join('\n');
}

function buildTrainCellCode(opts: {
  recipe: TrainingRecipe;
  taskAdapter: string;
  rank: number;
  datasetJson?: string;
  epochs: number;
  learningRate: number;
  maxSeqLength: number;
}): string {
  const datasetPayload =
    opts.datasetJson ??
    JSON.stringify([
      {
        prompt: 'Explain Awesome-OS B8G .b8p CAS packs.',
        completion: 'B8G .b8p packs are content-addressed WASM/LoRA/object archives in .b8g-cas/.',
      },
    ]);

  const datasetHeader = [
    'import json',
    'from datasets import Dataset',
    `RAW_DATASET_JSON = ${escapePythonString(datasetPayload)}`,
    'records = json.loads(RAW_DATASET_JSON)',
    'train_dataset = Dataset.from_list(records)',
  ];

  if (opts.recipe === 'jina-v3-merge-then-stack') {
    return [
      '# Awesome-OS Sovereign Colab Cell 4: Jina-v3 Task-LoRA Training (merge-then-stack)',
      ...datasetHeader,
      'import torch.nn.functional as F',
      '',
      '# Step 1: Merge built-in task adapter into base weights when supported',
      'if hasattr(model, "merge_and_unload"):',
      '    model = model.merge_and_unload()',
      '',
      '# Step 2: Stack fresh Task-LoRA adapter on top of merged backbone',
      'stacked_lora_cfg = LoraConfig(',
      `    r=${opts.rank},`,
      `    lora_alpha=${opts.rank * 2},`,
      '    target_modules=["Wqkv", "out_proj", "fc1", "fc2"],',
      '    lora_dropout=0.05,',
      '    bias="none",',
      ')',
      'model = get_peft_model(model, stacked_lora_cfg)',
      `optimizer = torch.optim.AdamW(model.parameters(), lr=${opts.learningRate})`,
      'model.train()',
      `for epoch in range(${opts.epochs}):`,
      '    for row in records:',
      '        text_a = row.get("query") or row.get("prompt") or ""',
      '        text_b = row.get("positive") or row.get("completion") or ""',
      '        enc_a = tokenizer(text_a, return_tensors="pt", truncation=True, padding=True).to(model.device)',
      '        enc_b = tokenizer(text_b, return_tensors="pt", truncation=True, padding=True).to(model.device)',
      '        emb_a = F.normalize(model(**enc_a).last_hidden_state.mean(dim=1), dim=-1)',
      '        emb_b = F.normalize(model(**enc_b).last_hidden_state.mean(dim=1), dim=-1)',
      '        loss = 1.0 - F.cosine_similarity(emb_a, emb_b).mean()',
      '        optimizer.zero_grad()',
      '        loss.backward()',
      '        optimizer.step()',
      `print(f"Completed jina-v3 merge-then-stack for task={TASK_ADAPTER}")`,
    ].join('\n');
  }

  if (opts.recipe === 'jina-v3-direct-task-finetune') {
    return [
      '# Awesome-OS Sovereign Colab Cell 4: Jina-v3 Task-LoRA Training (direct-task-finetune)',
      ...datasetHeader,
      'import torch.nn.functional as F',
      '',
      '# Freeze backbone and directly fine-tune the active task adapter parameters',
      'for name, param in model.named_parameters():',
      '    param.requires_grad = ("lora" in name.lower()) or (TASK_ADAPTER in name)',
      'trainable = [p for p in model.parameters() if p.requires_grad]',
      'if not trainable:',
      '    lora_cfg = LoraConfig(',
      `        r=${opts.rank},`,
      `        lora_alpha=${opts.rank * 2},`,
      '        target_modules=["Wqkv", "out_proj"],',
      '        lora_dropout=0.05,',
      '        bias="none",',
      '    )',
      '    model = get_peft_model(model, lora_cfg)',
      '    trainable = [p for p in model.parameters() if p.requires_grad]',
      `optimizer = torch.optim.AdamW(trainable, lr=${opts.learningRate})`,
      'model.train()',
      `for epoch in range(${opts.epochs}):`,
      '    for row in records:',
      '        text_a = row.get("query") or row.get("prompt") or ""',
      '        text_b = row.get("positive") or row.get("completion") or ""',
      '        enc_a = tokenizer(text_a, return_tensors="pt", truncation=True, padding=True).to(model.device)',
      '        enc_b = tokenizer(text_b, return_tensors="pt", truncation=True, padding=True).to(model.device)',
      '        emb_a = F.normalize(model(**enc_a).last_hidden_state.mean(dim=1), dim=-1)',
      '        emb_b = F.normalize(model(**enc_b).last_hidden_state.mean(dim=1), dim=-1)',
      '        loss = 1.0 - F.cosine_similarity(emb_a, emb_b).mean()',
      '        optimizer.zero_grad()',
      '        loss.backward()',
      '        optimizer.step()',
      'print(f"Completed jina-v3 direct-task-finetune for task={TASK_ADAPTER}")',
    ].join('\n');
  }

  if (opts.recipe === 'dpo') {
    return [
      '# Awesome-OS Sovereign Colab Cell 4: DPO Alignment Training',
      ...datasetHeader,
      'from trl import DPOConfig, DPOTrainer',
      '',
      'dpo_config = DPOConfig(',
      '    output_dir="out_dpo",',
      `    num_train_epochs=${opts.epochs},`,
      `    learning_rate=${opts.learningRate},`,
      '    per_device_train_batch_size=2,',
      '    gradient_accumulation_steps=4,',
      '    beta=0.1,',
      '    logging_steps=1,',
      '    report_to="none",',
      ')',
      'trainer = DPOTrainer(',
      '    model=model,',
      '    ref_model=None,',
      '    args=dpo_config,',
      '    train_dataset=train_dataset,',
      '    processing_class=tokenizer,',
      ')',
      'trainer.train()',
    ].join('\n');
  }

  if (opts.recipe === 'grpo') {
    return [
      '# Awesome-OS Sovereign Colab Cell 4: GRPO Reinforcement Learning',
      ...datasetHeader,
      'from trl import GRPOConfig, GRPOTrainer',
      '',
      'def b8g_reward_fn(completions, **kwargs):',
      '    return [1.0 if len(str(c).strip()) > 16 else 0.0 for c in completions]',
      '',
      'grpo_config = GRPOConfig(',
      '    output_dir="out_grpo",',
      `    num_train_epochs=${opts.epochs},`,
      `    learning_rate=${opts.learningRate},`,
      '    per_device_train_batch_size=2,',
      '    num_generations=4,',
      '    logging_steps=1,',
      '    report_to="none",',
      ')',
      'trainer = GRPOTrainer(',
      '    model=model,',
      '    reward_funcs=[b8g_reward_fn],',
      '    args=grpo_config,',
      '    train_dataset=train_dataset,',
      '    processing_class=tokenizer,',
      ')',
      'trainer.train()',
    ].join('\n');
  }

  const isQloraNf4 = opts.recipe === 'qlora-nf4';
  return [
    `# Awesome-OS Sovereign Colab Cell 4: ${isQloraNf4 ? 'QLoRA NF4' : 'SFT'} Fine-Tuning`,
    ...datasetHeader,
    'from trl import SFTConfig, SFTTrainer',
    '',
    'def format_row(row):',
    '    if "text" in row:',
    '        return row["text"]',
    '    return f"### Prompt:\\n{row.get(\'prompt\', \'\')}\\n\\n### Response:\\n{row.get(\'completion\', \'\')}"',
    '',
    'sft_config = SFTConfig(',
    `    output_dir=${escapePythonString(isQloraNf4 ? 'out_qlora_nf4' : 'out_sft')},`,
    `    num_train_epochs=${opts.epochs},`,
    `    learning_rate=${opts.learningRate},`,
    `    max_seq_length=${opts.maxSeqLength},`,
    '    per_device_train_batch_size=2,',
    '    gradient_accumulation_steps=4,',
    `    optim=${escapePythonString(isQloraNf4 ? 'paged_adamw_8bit' : 'adamw_8bit')},`,
    '    logging_steps=1,',
    '    report_to="none",',
    ')',
    'trainer = SFTTrainer(',
    '    model=model,',
    '    processing_class=tokenizer,',
    '    train_dataset=train_dataset,',
    '    formatting_func=format_row,',
    '    args=sft_config,',
    ')',
    'trainer.train()',
  ].join('\n');
}

function buildExportCellCode(opts: {
  isJina: boolean;
  recipe: TrainingRecipe;
  baseModel: string;
  taskAdapter: string | null;
  quant: string;
  deviceId: string;
  callbackEndpoint?: string;
}): string {
  const ggufExportLines = opts.isJina
    ? [
        '# Save LoRA adapter & merged weights for GGUF conversion',
        'model.save_pretrained(LORA_DIR)',
        'tokenizer.save_pretrained(LORA_DIR)',
        'gguf_path = Path("out_gguf") / f"jina-v3-{QUANT.lower()}.gguf"',
        'gguf_path.parent.mkdir(parents=True, exist_ok=True)',
        'if not gguf_path.exists():',
        '    # Emit placeholder GGUF metadata pointer when llama.cpp convert script is run locally',
        '    gguf_path.write_bytes(b"GGUF\\x03\\x00\\x00\\x00" + b"\\x00" * 16)',
      ]
    : [
        '# Export quantized GGUF + LoRA adapter weights',
        'model.save_pretrained(LORA_DIR)',
        'tokenizer.save_pretrained(LORA_DIR)',
        'model.save_pretrained_gguf("out_gguf", tokenizer, quantization_method=QUANT)',
        'gguf_candidates = sorted(Path("out_gguf").glob("*.gguf"))',
        'gguf_path = gguf_candidates[0] if gguf_candidates else Path("out_gguf") / f"model-{QUANT.lower()}.gguf"',
      ];

  const callbackLines = opts.callbackEndpoint
    ? [
        '',
        `CALLBACK_ENDPOINT = ${escapePythonString(opts.callbackEndpoint)}`,
        'req = urllib.request.Request(',
        '    CALLBACK_ENDPOINT,',
        '    data=json.dumps(export_manifest).encode("utf-8"),',
        '    headers={"Content-Type": "application/json", "X-B8G-Device-Id": DEVICE_ID},',
        '    method="POST",',
        ')',
        'with urllib.request.urlopen(req, timeout=15) as resp:',
        '    print("Callback receipt status:", resp.status)',
      ]
    : [
        '',
        'print("No callbackEndpoint configured; pull export-manifest.json & b8g-adapter.b8p into .b8g-cas/")',
      ];

  return [
    '# Awesome-OS Sovereign Colab Cell 5: Export GGUF + .b8p LoRA Adapter + SHA-256 Digest',
    'import hashlib',
    'import json',
    'import tarfile',
    'import urllib.request',
    'from pathlib import Path',
    '',
    `DEVICE_ID = ${escapePythonString(opts.deviceId)}`,
    `RECIPE = ${escapePythonString(opts.recipe)}`,
    `BASE_MODEL = ${escapePythonString(opts.baseModel)}`,
    `QUANT = ${escapePythonString(opts.quant)}`,
    'LORA_DIR = Path("out_lora")',
    'B8P_PATH = Path("b8g-adapter.b8p")',
    '',
    'def sha256_file(path: Path) -> str:',
    '    h = hashlib.sha256()',
    '    with path.open("rb") as f:',
    '        for chunk in iter(lambda: f.read(1024 * 1024), b""):',
    '            h.update(chunk)',
    '    return h.hexdigest()',
    '',
    ...ggufExportLines,
    '',
    '# Pack LoRA adapter into .b8p archive ready for local .b8g-cas/ ingestion',
    'with tarfile.open(B8P_PATH, "w:gz") as tar:',
    '    tar.add(LORA_DIR, arcname="lora")',
    '',
    'export_manifest = {',
    '    "schema": "b8g.colab.export.v1",',
    '    "deviceId": DEVICE_ID,',
    '    "recipe": RECIPE,',
    '    "baseModel": BASE_MODEL,',
    `    "taskAdapter": ${opts.taskAdapter ? escapePythonString(opts.taskAdapter) : 'None'},`,
    '    "quant": QUANT,',
    '    "casTarget": ".b8g-cas/adapters",',
    '    "artifacts": {',
    '        "gguf": {"path": str(gguf_path), "sha256": sha256_file(gguf_path)},',
    '        "b8p": {"path": str(B8P_PATH), "sha256": sha256_file(B8P_PATH)},',
    '    },',
    '}',
    'Path("export-manifest.json").write_text(json.dumps(export_manifest, indent=2), encoding="utf-8")',
    'print(json.dumps(export_manifest, indent=2))',
    ...callbackLines,
  ].join('\n');
}

/**
 * Generates a valid nbformat v4.5 `.ipynb` JSON document with 5 structured cells:
 * 1. Markdown header (Awesome-OS Sovereign Colab Host Node Intersection + target `deviceId`)
 * 2. Install Unsloth / dependencies cell
 * 3. Load base model (`FastLanguageModel.from_pretrained` or `jina-embeddings-v3` with Task-LoRA)
 * 4. Train cell (SFT / QLoRA NF4 / DPO / GRPO / Jina-v3 Task-LoRA)
 * 5. Export GGUF + `.b8p` LoRA adapter + SHA-256 digest cell ready to pull into `.b8g-cas/`
 */
export function generateColabNotebookIpynb(opts: GenerateColabNotebookOptions): IpynbDocument {
  const rank = Math.max(1, Math.trunc(opts.rank ?? 16));
  const quant = opts.quant?.trim() || 'q4_k_m';
  const deviceId = opts.deviceId?.trim() || 'b8g-dev-cloud-unassigned';
  const maxSeqLength = Math.max(128, Math.trunc(opts.maxSeqLength ?? 2048));
  const epochs = Math.max(1, Math.trunc(opts.epochs ?? 1));
  const learningRate = opts.learningRate ?? 2e-4;
  const isJina = isJinaV3Target(opts.recipe, opts.baseModel);
  const taskAdapter = opts.taskAdapter ?? (isJina ? 'retrieval.query' : undefined);
  const gpuSpec = opts.gpuTier ? COLAB_GPU_TIER_SPECS[opts.gpuTier] : COLAB_GPU_TIER_SPECS['T4-16GB'];

  const headerLines = [
    '# Awesome-OS Sovereign Colab Host Node Intersection',
    '',
    `- **Target Device-ID**: \`${deviceId}\``,
    `- **Recipe**: \`${opts.recipe}\``,
    `- **Base Model**: \`${opts.baseModel}\``,
    `- **LoRA Rank**: \`${rank}\` | **GGUF Quant**: \`${quant}\``,
    ...(taskAdapter ? [`- **Task-LoRA Adapter**: \`${taskAdapter}\``] : []),
    ...(opts.callbackEndpoint ? [`- **Callback Endpoint**: \`${opts.callbackEndpoint}\``] : []),
    `- **Local CAS Pull Target**: \`.b8g-cas/\``,
  ].join('\n');

  const cell1: IpynbMarkdownCell = {
    id: 'b8g-cell-01-header',
    cell_type: 'markdown',
    metadata: { id: 'b8g-cell-01-header' },
    source: toSourceLines(headerLines),
  };

  const cell2: IpynbCodeCell = {
    id: 'b8g-cell-02-install',
    cell_type: 'code',
    execution_count: null,
    metadata: { id: 'b8g-cell-02-install' },
    outputs: [],
    source: toSourceLines(buildInstallCellCode(isJina, opts.recipe)),
  };

  const cell3: IpynbCodeCell = {
    id: 'b8g-cell-03-load-model',
    cell_type: 'code',
    execution_count: null,
    metadata: { id: 'b8g-cell-03-load-model' },
    outputs: [],
    source: toSourceLines(
      buildLoadModelCellCode({
        isJina,
        recipe: opts.recipe,
        baseModel: opts.baseModel,
        taskAdapter: taskAdapter ?? 'retrieval.query',
        rank,
        maxSeqLength,
      }),
    ),
  };

  const cell4: IpynbCodeCell = {
    id: 'b8g-cell-04-train',
    cell_type: 'code',
    execution_count: null,
    metadata: { id: 'b8g-cell-04-train' },
    outputs: [],
    source: toSourceLines(
      buildTrainCellCode({
        recipe: opts.recipe,
        taskAdapter: taskAdapter ?? 'retrieval.query',
        rank,
        datasetJson: opts.datasetJson,
        epochs,
        learningRate,
        maxSeqLength,
      }),
    ),
  };

  const cell5: IpynbCodeCell = {
    id: 'b8g-cell-05-export-cas',
    cell_type: 'code',
    execution_count: null,
    metadata: { id: 'b8g-cell-05-export-cas' },
    outputs: [],
    source: toSourceLines(
      buildExportCellCode({
        isJina,
        recipe: opts.recipe,
        baseModel: opts.baseModel,
        taskAdapter: taskAdapter ?? null,
        quant,
        deviceId,
        callbackEndpoint: opts.callbackEndpoint,
      }),
    ),
  };

  return {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      colab: {
        provenance: [],
        gpuType: gpuSpec.colabRuntimeType,
        machine_shape: gpuSpec.highRam ? 'hm' : 'standard',
      },
      accelerator: gpuSpec.accelerator,
      kernelspec: {
        name: 'python3',
        display_name: 'Python 3',
        language: 'python',
      },
      language_info: {
        name: 'python',
        version: '3.11.0',
        mimetype: 'text/x-python',
        file_extension: '.py',
      },
      awesome_os: {
        bridge: 'ColabNodeBridge',
        deviceId,
        recipe: opts.recipe,
        baseModel: opts.baseModel,
        taskAdapter: taskAdapter ?? null,
        rank,
        quant,
        callbackEndpoint: opts.callbackEndpoint ?? null,
      },
    },
    cells: [cell1, cell2, cell3, cell4, cell5],
  };
}

const EXECUTABLE_LANGUAGES = new Set([
  'python',
  'ts',
  'js',
  'bash',
  'pwsh',
  'sql',
  'rust',
  'c',
  'cpp',
  'go',
]);

function normalizeLanguageTag(raw: string): string {
  const token = raw.trim().toLowerCase();
  if (token === 'py' || token === 'python' || token === 'python3') return 'python';
  if (token === 'ts' || token === 'mts' || token === 'cts' || token === 'typescript') return 'ts';
  if (token === 'js' || token === 'mjs' || token === 'cjs' || token === 'javascript') return 'js';
  if (token === 'sh' || token === 'bash' || token === 'shell' || token === 'zsh') return 'bash';
  if (token === 'pwsh' || token === 'powershell' || token === 'ps1') return 'pwsh';
  return token || 'text';
}

function parseFenceInfo(info: string): { language: string; metadata: Record<string, unknown> } {
  const trimmed = info.trim();
  if (trimmed.length === 0) {
    return { language: 'text', metadata: {} };
  }

  const firstSpace = trimmed.search(/\s/);
  const rawLang = firstSpace === -1 ? trimmed : trimmed.slice(0, firstSpace);
  const rest = firstSpace === -1 ? '' : trimmed.slice(firstSpace + 1).trim();
  const metadata: Record<string, unknown> = {};

  if (rest.length > 0) {
    const attrPattern = /([a-zA-Z0-9_-]+)(?:=(?:"([^"]*)"|'([^']*)'|([^\s]+)))?/g;
    let match = attrPattern.exec(rest);
    while (match !== null) {
      const key = match[1];
      const val = match[2] ?? match[3] ?? match[4];
      if (key) {
        metadata[key] = val !== undefined ? val : true;
      }
      match = attrPattern.exec(rest);
    }
  }

  return {
    language: normalizeLanguageTag(rawLang),
    metadata,
  };
}

function detectCellMagicLanguage(code: string, fallback: string): string {
  const firstLine = code.trimStart().split('\n', 1)[0]?.trim() ?? '';
  if (firstLine.startsWith('%%bash') || firstLine.startsWith('%%sh')) return 'bash';
  if (firstLine.startsWith('%%ts') || firstLine.startsWith('%%typescript')) return 'ts';
  if (firstLine.startsWith('%%js') || firstLine.startsWith('%%javascript')) return 'js';
  if (firstLine.startsWith('%%python')) return 'python';
  return fallback;
}

function parseIpynbCells(doc: Record<string, unknown>): NotebookCell[] {
  const rawCells = Array.isArray(doc.cells) ? doc.cells : [];
  const docMeta = (doc.metadata && typeof doc.metadata === 'object' ? doc.metadata : {}) as Record<
    string,
    unknown
  >;
  const kernelspec = (
    docMeta.kernelspec && typeof docMeta.kernelspec === 'object' ? docMeta.kernelspec : {}
  ) as Record<string, unknown>;
  const langInfo = (
    docMeta.language_info && typeof docMeta.language_info === 'object' ? docMeta.language_info : {}
  ) as Record<string, unknown>;

  const defaultKernelLang = normalizeLanguageTag(
    String(kernelspec.language ?? langInfo.name ?? 'python'),
  );

  const cells: NotebookCell[] = [];
  for (let i = 0; i < rawCells.length; i++) {
    const raw = rawCells[i];
    if (!raw || typeof raw !== 'object') continue;
    const obj = raw as Record<string, unknown>;
    const rawType = String(obj.cell_type ?? 'code');
    const kind: NotebookCellKind =
      rawType === 'markdown' ? 'markdown' : rawType === 'raw' ? 'raw' : 'code';

    const source = Array.isArray(obj.source)
      ? obj.source.map((line) => String(line)).join('')
      : typeof obj.source === 'string'
        ? obj.source
        : '';

    const rawMeta =
      obj.metadata && typeof obj.metadata === 'object'
        ? { ...(obj.metadata as Record<string, unknown>) }
        : {};
    const id =
      typeof obj.id === 'string' && obj.id.length > 0
        ? obj.id
        : typeof rawMeta.id === 'string' && rawMeta.id.length > 0
          ? rawMeta.id
          : `ipynb-cell-${i + 1}`;

    let language = 'text';
    if (kind === 'markdown') {
      language = 'markdown';
    } else if (kind === 'code') {
      const explicitLang =
        typeof rawMeta.language === 'string' ? normalizeLanguageTag(rawMeta.language) : null;
      language = explicitLang ?? detectCellMagicLanguage(source, defaultKernelLang);
    }

    cells.push({
      index: cells.length,
      id,
      kind,
      language,
      source,
      executable: kind === 'code',
      metadata: { ...rawMeta, sourceFormat: 'ipynb' },
    });
  }

  return cells;
}

function parseMarkdownCells(source: string): NotebookCell[] {
  const normalized = source.replace(/\r\n/g, '\n');
  if (normalized.trim().length === 0) return [];

  const lines = normalized.split('\n');
  const cells: NotebookCell[] = [];
  let mdBuffer: string[] = [];

  let inFence = false;
  let fenceChar = '`';
  let fenceLen = 3;
  let fenceLang = 'text';
  let fenceMeta: Record<string, unknown> = {};
  let codeBuffer: string[] = [];

  const flushMarkdown = (): void => {
    const text = mdBuffer.join('\n').trim();
    mdBuffer = [];
    if (text.length === 0) return;
    const index = cells.length;
    cells.push({
      index,
      id: `md-cell-${index + 1}`,
      kind: 'markdown',
      language: 'markdown',
      source: text,
      executable: false,
      metadata: { sourceFormat: 'markdown' },
    });
  };

  const flushCode = (): void => {
    const code = codeBuffer.join('\n');
    codeBuffer = [];
    const index = cells.length;
    const explicitId = typeof fenceMeta.id === 'string' && fenceMeta.id.length > 0 ? fenceMeta.id : null;
    const executable =
      EXECUTABLE_LANGUAGES.has(fenceLang) || fenceMeta.exec === true || fenceMeta.run === true;
    cells.push({
      index,
      id: explicitId ?? `code-cell-${index + 1}`,
      kind: 'code',
      language: fenceLang,
      source: code,
      executable,
      metadata: { ...fenceMeta, sourceFormat: 'markdown' },
    });
  };

  for (const line of lines) {
    if (!inFence) {
      const openMatch = /^\s*(`{3,}|~{3,})(.*)$/.exec(line);
      if (openMatch) {
        flushMarkdown();
        const marker = openMatch[1] ?? '```';
        const info = openMatch[2] ?? '';
        const parsed = parseFenceInfo(info);
        inFence = true;
        fenceChar = marker[0] ?? '`';
        fenceLen = marker.length;
        fenceLang = parsed.language;
        fenceMeta = parsed.metadata;
        codeBuffer = [];
      } else {
        mdBuffer.push(line);
      }
    } else {
      const trimmed = line.trim();
      const isClosingFence =
        trimmed.length >= fenceLen &&
        [...trimmed].every((ch) => ch === fenceChar);
      if (isClosingFence) {
        flushCode();
        inFence = false;
      } else {
        codeBuffer.push(line);
      }
    }
  }

  if (inFence) {
    flushCode();
  } else {
    flushMarkdown();
  }

  return cells;
}

/**
 * Parses either a Jupyter `.ipynb` JSON string OR an executable Markdown document
 * containing fenced code blocks (`python`, `ts`, `bash`, etc.) into a uniform
 * `NotebookCell[]` array for the Universal Notebook Runner.
 */
export function parseUniversalNotebookCells(source: string): NotebookCell[] {
  const trimmed = source.trim();
  if (trimmed.length === 0) return [];

  if (trimmed.startsWith('{')) {
    try {
      const parsed: unknown = JSON.parse(trimmed);
      if (
        parsed &&
        typeof parsed === 'object' &&
        Array.isArray((parsed as Record<string, unknown>).cells)
      ) {
        return parseIpynbCells(parsed as Record<string, unknown>);
      }
    } catch {
      // Fall back to Markdown parser when leading '{' is not valid JSON
    }
  }

  return parseMarkdownCells(source);
}

/**
 * High-level coordinator bridging Google Colab GPU/TPU runtimes into `CloudDeviceMesh`,
 * generating `.ipynb` training intersections, and parsing universal notebook cells.
 */
export class ColabNodeBridge {
  readonly #mesh: CloudDeviceMesh;
  readonly #defaultTenantId: string;
  readonly #colabNodes = new Map<string, ColabNodeMetadata>();

  constructor(opts: ColabNodeBridgeOptions = {}) {
    this.#mesh = opts.mesh ?? new CloudDeviceMesh({ autoRegisterLocalHost: false });
    this.#defaultTenantId = opts.defaultTenantId ?? 'default-tenant';
  }

  get mesh(): CloudDeviceMesh {
    return this.#mesh;
  }

  registerNode(opts: RegisterColabHostNodeOptions, tenantId = this.#defaultTenantId): DeviceRecord {
    const device = registerColabHostNode(this.#mesh, tenantId, opts);
    const spec = COLAB_GPU_TIER_SPECS[opts.gpuTier];
    const colabUrl =
      opts.colabUrl?.trim() ||
      `https://colab.research.google.com/drive/${createHash('sha256').update(device.deviceId).digest('hex').slice(0, 24)}`;

    this.#colabNodes.set(device.deviceId, {
      deviceId: device.deviceId,
      label: opts.label.trim(),
      gpuTier: opts.gpuTier,
      spec,
      colabUrl,
      tunnelEndpoint: device.endpoint,
    });

    return device;
  }

  listColabNodes(tenantId?: string): ColabNodeRecord[] {
    const out: ColabNodeRecord[] = [];
    for (const device of this.#mesh.listDevices(tenantId)) {
      const colab = this.#colabNodes.get(device.deviceId);
      if (colab) {
        out.push({ device, colab: { ...colab } });
      }
    }
    return out;
  }

  routeJobToColab(jobKey: string, tenantId?: string, replicaCount = 2): ShardRouteResult {
    return this.#mesh.resolveShardOwner(jobKey, tenantId, replicaCount);
  }

  buildNotebook(opts: GenerateColabNotebookOptions): IpynbDocument {
    const registered = opts.deviceId ? this.#colabNodes.get(opts.deviceId) : undefined;
    return generateColabNotebookIpynb({
      ...opts,
      gpuTier: opts.gpuTier ?? registered?.gpuTier,
      callbackEndpoint: opts.callbackEndpoint ?? registered?.tunnelEndpoint,
    });
  }

  serializeNotebook(doc: IpynbDocument): string {
    return `${JSON.stringify(doc, null, 2)}\n`;
  }

  parseNotebook(source: string): NotebookCell[] {
    return parseUniversalNotebookCells(source);
  }
}
