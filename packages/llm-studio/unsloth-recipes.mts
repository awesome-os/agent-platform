import { createHash } from 'node:crypto';

export type UnslothRecipeId =
  | 'sft-lora'
  | 'qlora-nf4'
  | 'dpo'
  | 'grpo'
  | 'orpo'
  | 'kto'
  | 'embed-jina-v3-lora';

export type UnslothRecipeCategory =
  | 'supervised'
  | 'preference'
  | 'rl-reasoning'
  | 'embedding';

export type DatasetFormat =
  | 'alpaca'
  | 'sharegpt'
  | 'chatml'
  | 'preference-dpo'
  | 'contrastive-triplet';

export type LoraTargetPresetId =
  | 'attention-only'
  | 'all-linear'
  | 'jina-v3-task';

export type GgufQuantId =
  | 'UD-Q4_K_XL'
  | 'Q4_K_M'
  | 'Q5_K_M'
  | 'Q6_K'
  | 'Q8_0'
  | 'F16';

export type EmbedAdapterStrategy =
  | 'merge-then-stack'
  | 'direct-task-finetune';

export interface UnslothRecipeSpec {
  readonly id: UnslothRecipeId;
  readonly name: string;
  readonly category: UnslothRecipeCategory;
  readonly description: string;
  readonly supportedDatasets: readonly DatasetFormat[];
  readonly defaultTargetPreset: LoraTargetPresetId;
  readonly defaultRank: number;
  readonly defaultAlpha: number;
  readonly defaultDropout: number;
  readonly defaultLearningRate: number;
  readonly loadIn4Bit: boolean;
  readonly dynamicQuantization2_0: boolean;
  readonly preservedHighPrecisionLayers: readonly string[];
  readonly sequencePacking: boolean;
  readonly useGradientCheckpointing: 'unsloth' | boolean;
  readonly vramMultiplierVsFp16: number;
  readonly speedupMultiplier: number;
  readonly trainerClass: string;
  readonly hyperparameters: Readonly<
    Record<string, string | number | boolean | readonly string[] | readonly number[]>
  >;
}

export interface GgufQuantSpec {
  readonly id: GgufQuantId;
  readonly label: string;
  readonly bitsPerWeight: number;
  readonly isDynamic2_0: boolean;
  readonly qualityTier:
    | 'dynamic-optimal'
    | 'balanced'
    | 'high'
    | 'very-high'
    | 'near-lossless'
    | 'lossless-fp16';
  readonly description: string;
  readonly estimateSizeMb: (paramBillions: number) => number;
}

export const LORA_TARGET_PRESETS: Readonly<Record<LoraTargetPresetId, readonly string[]>> = Object.freeze({
  'attention-only': Object.freeze(['q_proj', 'k_proj', 'v_proj', 'o_proj']),
  'all-linear': Object.freeze([
    'q_proj',
    'k_proj',
    'v_proj',
    'o_proj',
    'gate_proj',
    'up_proj',
    'down_proj',
  ]),
  'jina-v3-task': Object.freeze(['wqkv', 'wo', 'gated_layers']),
});

const BYTES_PER_MIB = 1024 * 1024;
const GGUF_HEADER_OVERHEAD_FACTOR = 1.015;

export function estimateGgufSizeMb(paramBillions: number, quant: GgufQuantId): number {
  if (!Number.isFinite(paramBillions) || paramBillions <= 0) {
    throw new RangeError(`paramBillions must be a positive finite number, received ${String(paramBillions)}`);
  }
  const spec = GGUF_EXPORT_QUANTS[quant];
  if (!spec) {
    throw new RangeError(`Unknown GGUF quantization preset: ${String(quant)}`);
  }
  const rawBytes = paramBillions * 1_000_000_000 * (spec.bitsPerWeight / 8);
  const totalMb = (rawBytes * GGUF_HEADER_OVERHEAD_FACTOR) / BYTES_PER_MIB;
  return Math.round(totalMb * 100) / 100;
}

export const GGUF_EXPORT_QUANTS: Readonly<Record<GgufQuantId, GgufQuantSpec>> = Object.freeze({
  'UD-Q4_K_XL': Object.freeze({
    id: 'UD-Q4_K_XL',
    label: 'Unsloth Dynamic 2.0 Q4_K_XL',
    bitsPerWeight: 4.85,
    isDynamic2_0: true,
    qualityTier: 'dynamic-optimal',
    description:
      'Unsloth Dynamic 2.0 4-bit quant preserving sensitive attention, MoE router, and output layers in higher precision.',
    estimateSizeMb: (paramBillions: number) => estimateGgufSizeMb(paramBillions, 'UD-Q4_K_XL'),
  }),
  Q4_K_M: Object.freeze({
    id: 'Q4_K_M',
    label: '4-bit K-Quant Medium',
    bitsPerWeight: 4.83,
    isDynamic2_0: false,
    qualityTier: 'balanced',
    description: 'Standard 4-bit K-quant offering strong speed/quality balance for local inference.',
    estimateSizeMb: (paramBillions: number) => estimateGgufSizeMb(paramBillions, 'Q4_K_M'),
  }),
  Q5_K_M: Object.freeze({
    id: 'Q5_K_M',
    label: '5-bit K-Quant Medium',
    bitsPerWeight: 5.69,
    isDynamic2_0: false,
    qualityTier: 'high',
    description: '5-bit K-quant with lower perplexity loss for coding and instruction following.',
    estimateSizeMb: (paramBillions: number) => estimateGgufSizeMb(paramBillions, 'Q5_K_M'),
  }),
  Q6_K: Object.freeze({
    id: 'Q6_K',
    label: '6-bit K-Quant',
    bitsPerWeight: 6.57,
    isDynamic2_0: false,
    qualityTier: 'very-high',
    description: 'High-precision 6-bit K-quant with negligible degradation relative to FP16.',
    estimateSizeMb: (paramBillions: number) => estimateGgufSizeMb(paramBillions, 'Q6_K'),
  }),
  Q8_0: Object.freeze({
    id: 'Q8_0',
    label: '8-bit Uniform Quant',
    bitsPerWeight: 8.5,
    isDynamic2_0: false,
    qualityTier: 'near-lossless',
    description: 'Near-lossless 8-bit quantization ideal for adapter merging and evaluation baselines.',
    estimateSizeMb: (paramBillions: number) => estimateGgufSizeMb(paramBillions, 'Q8_0'),
  }),
  F16: Object.freeze({
    id: 'F16',
    label: '16-bit Half Precision',
    bitsPerWeight: 16.0,
    isDynamic2_0: false,
    qualityTier: 'lossless-fp16',
    description: 'Full FP16 weights for archival storage, re-quantization, or high-VRAM serving.',
    estimateSizeMb: (paramBillions: number) => estimateGgufSizeMb(paramBillions, 'F16'),
  }),
});

export const UNSLOTH_RECIPES: Readonly<Record<UnslothRecipeId, UnslothRecipeSpec>> = Object.freeze({
  'sft-lora': Object.freeze({
    id: 'sft-lora',
    name: 'SFT-LoRA (Sequence Packing + Unsloth Checkpointing)',
    category: 'supervised',
    description:
      'Supervised Fine-Tuning with sample sequence packing and Unsloth smart gradient checkpointing for 2x throughput and 60% VRAM reduction.',
    supportedDatasets: Object.freeze(['alpaca', 'sharegpt', 'chatml'] as const),
    defaultTargetPreset: 'all-linear',
    defaultRank: 16,
    defaultAlpha: 16,
    defaultDropout: 0,
    defaultLearningRate: 2e-4,
    loadIn4Bit: false,
    dynamicQuantization2_0: false,
    preservedHighPrecisionLayers: Object.freeze([]),
    sequencePacking: true,
    useGradientCheckpointing: 'unsloth',
    vramMultiplierVsFp16: 0.42,
    speedupMultiplier: 2.1,
    trainerClass: 'SFTTrainer',
    hyperparameters: Object.freeze({
      packing: true,
      use_gradient_checkpointing: 'unsloth',
      optim: 'adamw_8bit',
      lr_scheduler_type: 'cosine',
      warmup_ratio: 0.05,
      weight_decay: 0.01,
    }),
  }),
  'qlora-nf4': Object.freeze({
    id: 'qlora-nf4',
    name: 'Dynamic 2.0 NF4 QLoRA',
    category: 'supervised',
    description:
      '4-bit Dynamic NF4 QLoRA using Unsloth Dynamic 2.0 quantization, keeping sensitive layers (embeddings, norms, routers, lm_head) in higher precision.',
    supportedDatasets: Object.freeze(['alpaca', 'sharegpt', 'chatml'] as const),
    defaultTargetPreset: 'all-linear',
    defaultRank: 16,
    defaultAlpha: 32,
    defaultDropout: 0,
    defaultLearningRate: 2e-4,
    loadIn4Bit: true,
    dynamicQuantization2_0: true,
    preservedHighPrecisionLayers: Object.freeze([
      'embed_tokens',
      'lm_head',
      'input_layernorm',
      'post_attention_layernorm',
      'router.gate',
    ]),
    sequencePacking: true,
    useGradientCheckpointing: 'unsloth',
    vramMultiplierVsFp16: 0.28,
    speedupMultiplier: 2.3,
    trainerClass: 'SFTTrainer',
    hyperparameters: Object.freeze({
      load_in_4bit: true,
      bnb_4bit_quant_type: 'nf4',
      bnb_4bit_use_double_quant: true,
      dynamic_quant_2_0: true,
      packing: true,
      use_gradient_checkpointing: 'unsloth',
      optim: 'adamw_8bit',
    }),
  }),
  dpo: Object.freeze({
    id: 'dpo',
    name: 'Direct Preference Optimization (DPO)',
    category: 'preference',
    description:
      'Aligns model responses directly from chosen/rejected preference pairs without training a separate reward model.',
    supportedDatasets: Object.freeze(['preference-dpo'] as const),
    defaultTargetPreset: 'all-linear',
    defaultRank: 16,
    defaultAlpha: 16,
    defaultDropout: 0,
    defaultLearningRate: 5e-5,
    loadIn4Bit: true,
    dynamicQuantization2_0: true,
    preservedHighPrecisionLayers: Object.freeze(['embed_tokens', 'lm_head']),
    sequencePacking: false,
    useGradientCheckpointing: 'unsloth',
    vramMultiplierVsFp16: 0.36,
    speedupMultiplier: 1.9,
    trainerClass: 'DPOTrainer',
    hyperparameters: Object.freeze({
      beta: 0.1,
      loss_type: 'sigmoid',
      max_prompt_length: 1024,
      max_length: 2048,
      use_gradient_checkpointing: 'unsloth',
      optim: 'adamw_8bit',
    }),
  }),
  grpo: Object.freeze({
    id: 'grpo',
    name: 'Group Relative Policy Optimization (GRPO)',
    category: 'rl-reasoning',
    description:
      'DeepSeek-R1 style reinforcement learning using group-normalized verifiable reward functions without a critic network.',
    supportedDatasets: Object.freeze(['alpaca', 'chatml'] as const),
    defaultTargetPreset: 'all-linear',
    defaultRank: 32,
    defaultAlpha: 32,
    defaultDropout: 0,
    defaultLearningRate: 5e-6,
    loadIn4Bit: true,
    dynamicQuantization2_0: true,
    preservedHighPrecisionLayers: Object.freeze(['embed_tokens', 'lm_head']),
    sequencePacking: false,
    useGradientCheckpointing: 'unsloth',
    vramMultiplierVsFp16: 0.38,
    speedupMultiplier: 2.0,
    trainerClass: 'GRPOTrainer',
    hyperparameters: Object.freeze({
      num_generations: 8,
      max_prompt_length: 512,
      max_completion_length: 1024,
      beta: 0.04,
      reward_functions: Object.freeze([
        'xmlcount_reward_func',
        'strict_format_reward_func',
        'correctness_reward_func',
      ]),
      use_gradient_checkpointing: 'unsloth',
      optim: 'paged_adamw_8bit',
    }),
  }),
  orpo: Object.freeze({
    id: 'orpo',
    name: 'Odds-Ratio Preference Optimization (ORPO)',
    category: 'preference',
    description:
      'Monolithic reference-model-free preference optimization combining SFT negative log-likelihood with an odds-ratio penalty.',
    supportedDatasets: Object.freeze(['preference-dpo'] as const),
    defaultTargetPreset: 'all-linear',
    defaultRank: 16,
    defaultAlpha: 16,
    defaultDropout: 0,
    defaultLearningRate: 8e-5,
    loadIn4Bit: true,
    dynamicQuantization2_0: true,
    preservedHighPrecisionLayers: Object.freeze(['embed_tokens', 'lm_head']),
    sequencePacking: false,
    useGradientCheckpointing: 'unsloth',
    vramMultiplierVsFp16: 0.32,
    speedupMultiplier: 2.0,
    trainerClass: 'ORPOTrainer',
    hyperparameters: Object.freeze({
      beta: 0.1,
      max_prompt_length: 1024,
      max_length: 2048,
      use_gradient_checkpointing: 'unsloth',
      optim: 'adamw_8bit',
    }),
  }),
  kto: Object.freeze({
    id: 'kto',
    name: 'Kahneman-Tversky Optimization (KTO)',
    category: 'preference',
    description:
      'Prospect-theory preference alignment supporting unpaired binary desirable/undesirable feedback and preference pairs.',
    supportedDatasets: Object.freeze(['preference-dpo'] as const),
    defaultTargetPreset: 'all-linear',
    defaultRank: 16,
    defaultAlpha: 16,
    defaultDropout: 0,
    defaultLearningRate: 5e-5,
    loadIn4Bit: true,
    dynamicQuantization2_0: true,
    preservedHighPrecisionLayers: Object.freeze(['embed_tokens', 'lm_head']),
    sequencePacking: false,
    useGradientCheckpointing: 'unsloth',
    vramMultiplierVsFp16: 0.35,
    speedupMultiplier: 1.9,
    trainerClass: 'KTOTrainer',
    hyperparameters: Object.freeze({
      beta: 0.1,
      desirable_weight: 1.0,
      undesirable_weight: 1.0,
      use_gradient_checkpointing: 'unsloth',
      optim: 'adamw_8bit',
    }),
  }),
  'embed-jina-v3-lora': Object.freeze({
    id: 'embed-jina-v3-lora',
    name: 'Jina-v3 / ModernBERT / lumiBERT Task-Adapter LoRA',
    category: 'embedding',
    description:
      'Contrastive triplet embedding LoRA supporting merge-then-stack or direct-task-finetune strategies with Matryoshka Representation Learning (MRL) truncation.',
    supportedDatasets: Object.freeze(['contrastive-triplet'] as const),
    defaultTargetPreset: 'jina-v3-task',
    defaultRank: 8,
    defaultAlpha: 16,
    defaultDropout: 0.05,
    defaultLearningRate: 1e-4,
    loadIn4Bit: false,
    dynamicQuantization2_0: false,
    preservedHighPrecisionLayers: Object.freeze(['embeddings', 'norm']),
    sequencePacking: false,
    useGradientCheckpointing: 'unsloth',
    vramMultiplierVsFp16: 0.45,
    speedupMultiplier: 2.4,
    trainerClass: 'ContrastiveLoRATrainer',
    hyperparameters: Object.freeze({
      adapter_strategies: Object.freeze(['merge-then-stack', 'direct-task-finetune']),
      default_strategy: 'direct-task-finetune',
      matryoshka_dims: Object.freeze([1024, 512, 256, 128, 64, 32, 16]),
      task_adapters: Object.freeze([
        'retrieval.query',
        'retrieval.passage',
        'separation',
        'classification',
        'text-matching',
        'error-repair',
        'code-search',
      ]),
      loss_function: 'MultipleNegativesRankingLoss',
      temperature: 0.05,
    }),
  }),
});

export interface DatasetValidationError {
  readonly rowIndex: number;
  readonly field: string;
  readonly message: string;
}

export interface NormalizedAlpacaRow {
  readonly format: 'alpaca';
  readonly instruction: string;
  readonly input: string;
  readonly output: string;
  readonly estimatedTokens: number;
}

export interface NormalizedShareGptRow {
  readonly format: 'sharegpt';
  readonly conversations: readonly { readonly from: string; readonly value: string }[];
  readonly estimatedTokens: number;
}

export interface NormalizedChatMlRow {
  readonly format: 'chatml';
  readonly messages: readonly {
    readonly role: 'system' | 'user' | 'assistant' | 'tool';
    readonly content: string;
  }[];
  readonly estimatedTokens: number;
}

export interface NormalizedPreferenceDpoRow {
  readonly format: 'preference-dpo';
  readonly prompt: string;
  readonly chosen: string;
  readonly rejected: string;
  readonly estimatedTokens: number;
}

export interface NormalizedContrastiveTripletRow {
  readonly format: 'contrastive-triplet';
  readonly anchor: string;
  readonly positive: string;
  readonly negative: string;
  readonly estimatedTokens: number;
}

export type NormalizedDatasetRow =
  | NormalizedAlpacaRow
  | NormalizedShareGptRow
  | NormalizedChatMlRow
  | NormalizedPreferenceDpoRow
  | NormalizedContrastiveTripletRow;

export interface DatasetValidationResult {
  readonly valid: boolean;
  readonly format: DatasetFormat;
  readonly rowCount: number;
  readonly validRowCount: number;
  readonly tokenEstimate: number;
  readonly errors: readonly DatasetValidationError[];
  readonly normalizedRows: readonly NormalizedDatasetRow[];
}

const VALID_CHATML_ROLES = new Set(['system', 'user', 'assistant', 'tool']);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function estimateTextTokens(text: string): number {
  const trimmed = text.trim();
  if (trimmed.length === 0) {
    return 0;
  }
  const byChars = Math.ceil(trimmed.length / 4);
  const byWords = Math.ceil(trimmed.split(/\s+/).length * 1.3);
  return Math.max(1, Math.round((byChars + byWords) / 2));
}

export function validateDataset(
  format: DatasetFormat,
  rows: readonly unknown[],
): DatasetValidationResult {
  const errors: DatasetValidationError[] = [];
  const normalizedRows: NormalizedDatasetRow[] = [];

  if (!Array.isArray(rows) || rows.length === 0) {
    errors.push({
      rowIndex: -1,
      field: 'rows',
      message: 'Dataset must be a non-empty array of rows.',
    });
    return {
      valid: false,
      format,
      rowCount: 0,
      validRowCount: 0,
      tokenEstimate: 0,
      errors,
      normalizedRows,
    };
  }

  let totalTokens = 0;

  for (let i = 0; i < rows.length; i++) {
    const raw = rows[i];
    if (!isRecord(raw)) {
      errors.push({
        rowIndex: i,
        field: '$',
        message: 'Row must be a JSON object.',
      });
      continue;
    }

    if (format === 'alpaca') {
      const instruction = typeof raw.instruction === 'string' ? raw.instruction.trim() : '';
      const input = typeof raw.input === 'string' ? raw.input.trim() : '';
      const output = typeof raw.output === 'string' ? raw.output.trim() : '';
      let rowValid = true;

      if (!instruction) {
        errors.push({ rowIndex: i, field: 'instruction', message: 'Missing or empty instruction.' });
        rowValid = false;
      }
      if (raw.input !== undefined && typeof raw.input !== 'string') {
        errors.push({ rowIndex: i, field: 'input', message: 'Optional input field must be a string.' });
        rowValid = false;
      }
      if (!output) {
        errors.push({ rowIndex: i, field: 'output', message: 'Missing or empty output.' });
        rowValid = false;
      }

      if (rowValid) {
        const estimatedTokens =
          estimateTextTokens(instruction) + estimateTextTokens(input) + estimateTextTokens(output) + 6;
        totalTokens += estimatedTokens;
        normalizedRows.push({
          format: 'alpaca',
          instruction,
          input,
          output,
          estimatedTokens,
        });
      }
    } else if (format === 'sharegpt') {
      const conv = raw.conversations;
      if (!Array.isArray(conv) || conv.length === 0) {
        errors.push({
          rowIndex: i,
          field: 'conversations',
          message: 'ShareGPT row requires a non-empty conversations array.',
        });
        continue;
      }

      const turns: { from: string; value: string }[] = [];
      let rowValid = true;
      let rowTokens = 0;

      for (let t = 0; t < conv.length; t++) {
        const turn = conv[t];
        if (!isRecord(turn)) {
          errors.push({
            rowIndex: i,
            field: `conversations[${t}]`,
            message: 'Conversation turn must be an object.',
          });
          rowValid = false;
          continue;
        }
        const from = typeof turn.from === 'string' ? turn.from.trim() : '';
        const value = typeof turn.value === 'string' ? turn.value.trim() : '';
        if (!from) {
          errors.push({
            rowIndex: i,
            field: `conversations[${t}].from`,
            message: 'Conversation turn requires a non-empty "from" role.',
          });
          rowValid = false;
        }
        if (!value) {
          errors.push({
            rowIndex: i,
            field: `conversations[${t}].value`,
            message: 'Conversation turn requires a non-empty "value" string.',
          });
          rowValid = false;
        }
        if (from && value) {
          turns.push({ from, value });
          rowTokens += estimateTextTokens(value) + 4;
        }
      }

      if (rowValid) {
        totalTokens += rowTokens;
        normalizedRows.push({
          format: 'sharegpt',
          conversations: turns,
          estimatedTokens: rowTokens,
        });
      }
    } else if (format === 'chatml') {
      const msgs = raw.messages;
      if (!Array.isArray(msgs) || msgs.length === 0) {
        errors.push({
          rowIndex: i,
          field: 'messages',
          message: 'ChatML row requires a non-empty messages array.',
        });
        continue;
      }

      const normalizedMsgs: {
        role: 'system' | 'user' | 'assistant' | 'tool';
        content: string;
      }[] = [];
      let rowValid = true;
      let rowTokens = 0;

      for (let m = 0; m < msgs.length; m++) {
        const msg = msgs[m];
        if (!isRecord(msg)) {
          errors.push({
            rowIndex: i,
            field: `messages[${m}]`,
            message: 'ChatML message must be an object.',
          });
          rowValid = false;
          continue;
        }
        const role = typeof msg.role === 'string' ? msg.role.trim() : '';
        const content = typeof msg.content === 'string' ? msg.content.trim() : '';
        if (!VALID_CHATML_ROLES.has(role)) {
          errors.push({
            rowIndex: i,
            field: `messages[${m}].role`,
            message: `Invalid ChatML role "${role}"; expected system, user, assistant, or tool.`,
          });
          rowValid = false;
        }
        if (!content) {
          errors.push({
            rowIndex: i,
            field: `messages[${m}].content`,
            message: 'ChatML message requires non-empty content.',
          });
          rowValid = false;
        }
        if (VALID_CHATML_ROLES.has(role) && content) {
          normalizedMsgs.push({
            role: role as 'system' | 'user' | 'assistant' | 'tool',
            content,
          });
          rowTokens += estimateTextTokens(content) + 4;
        }
      }

      if (rowValid) {
        totalTokens += rowTokens;
        normalizedRows.push({
          format: 'chatml',
          messages: normalizedMsgs,
          estimatedTokens: rowTokens,
        });
      }
    } else if (format === 'preference-dpo') {
      const prompt = typeof raw.prompt === 'string' ? raw.prompt.trim() : '';
      const chosen = typeof raw.chosen === 'string' ? raw.chosen.trim() : '';
      const rejected = typeof raw.rejected === 'string' ? raw.rejected.trim() : '';
      let rowValid = true;

      if (!prompt) {
        errors.push({ rowIndex: i, field: 'prompt', message: 'Missing or empty prompt.' });
        rowValid = false;
      }
      if (!chosen) {
        errors.push({ rowIndex: i, field: 'chosen', message: 'Missing or empty chosen response.' });
        rowValid = false;
      }
      if (!rejected) {
        errors.push({ rowIndex: i, field: 'rejected', message: 'Missing or empty rejected response.' });
        rowValid = false;
      }
      if (chosen && rejected && chosen === rejected) {
        errors.push({
          rowIndex: i,
          field: 'rejected',
          message: 'Chosen and rejected responses must not be identical.',
        });
        rowValid = false;
      }

      if (rowValid) {
        const estimatedTokens =
          estimateTextTokens(prompt) + estimateTextTokens(chosen) + estimateTextTokens(rejected) + 8;
        totalTokens += estimatedTokens;
        normalizedRows.push({
          format: 'preference-dpo',
          prompt,
          chosen,
          rejected,
          estimatedTokens,
        });
      }
    } else if (format === 'contrastive-triplet') {
      const anchor = typeof raw.anchor === 'string' ? raw.anchor.trim() : '';
      const positive = typeof raw.positive === 'string' ? raw.positive.trim() : '';
      const negative = typeof raw.negative === 'string' ? raw.negative.trim() : '';
      let rowValid = true;

      if (!anchor) {
        errors.push({ rowIndex: i, field: 'anchor', message: 'Missing or empty anchor text.' });
        rowValid = false;
      }
      if (!positive) {
        errors.push({ rowIndex: i, field: 'positive', message: 'Missing or empty positive text.' });
        rowValid = false;
      }
      if (!negative) {
        errors.push({ rowIndex: i, field: 'negative', message: 'Missing or empty negative text.' });
        rowValid = false;
      }
      if (positive && negative && positive === negative) {
        errors.push({
          rowIndex: i,
          field: 'negative',
          message: 'Positive and negative texts must not be identical.',
        });
        rowValid = false;
      }

      if (rowValid) {
        const estimatedTokens =
          estimateTextTokens(anchor) + estimateTextTokens(positive) + estimateTextTokens(negative) + 6;
        totalTokens += estimatedTokens;
        normalizedRows.push({
          format: 'contrastive-triplet',
          anchor,
          positive,
          negative,
          estimatedTokens,
        });
      }
    } else {
      errors.push({
        rowIndex: i,
        field: 'format',
        message: `Unsupported dataset format: ${String(format)}`,
      });
    }
  }

  return {
    valid: errors.length === 0,
    format,
    rowCount: rows.length,
    validRowCount: normalizedRows.length,
    tokenEstimate: totalTokens,
    errors,
    normalizedRows,
  };
}

export function estimateTrainingVramGb(
  paramBillions: number,
  recipeId: UnslothRecipeId,
  maxSeqLength = 2048,
  batchSize = 2,
): number {
  if (!Number.isFinite(paramBillions) || paramBillions <= 0) {
    throw new RangeError(`paramBillions must be positive, received ${String(paramBillions)}`);
  }
  const recipe = UNSLOTH_RECIPES[recipeId];
  if (!recipe) {
    throw new RangeError(`Unknown recipe: ${String(recipeId)}`);
  }
  const fp16ModelGb = paramBillions * 2;
  const baseTrainingGb = fp16ModelGb * recipe.vramMultiplierVsFp16;
  const activationGb = (maxSeqLength / 2048) * batchSize * Math.max(0.35, paramBillions * 0.08);
  const overheadGb = 0.8;
  return Math.round((baseTrainingGb + activationGb + overheadGb) * 100) / 100;
}

export interface UnslothExportConfig {
  readonly modelId: string;
  readonly adapterName: string;
  readonly recipeId: UnslothRecipeId;
  readonly paramBillions: number;
  readonly quant: GgufQuantId;
  readonly datasetFormat?: DatasetFormat;
  readonly datasetPath?: string;
  readonly targetPreset?: LoraTargetPresetId;
  readonly customTargetModules?: readonly string[];
  readonly loraRank?: number;
  readonly loraAlpha?: number;
  readonly loraDropout?: number;
  readonly learningRate?: number;
  readonly maxSeqLength?: number;
  readonly epochs?: number;
  readonly batchSize?: number;
  readonly gradientAccumulationSteps?: number;
  readonly systemPrompt?: string;
  readonly temperature?: number;
  readonly contextLength?: number;
  readonly gpuLayers?: number;
  readonly serverPort?: number;
  readonly embedAdapterStrategy?: EmbedAdapterStrategy;
  readonly matryoshkaDims?: readonly number[];
  readonly includeGpuWorkerScript?: boolean;
  readonly cloudMeshNodeId?: string;
}

export interface B8pCasAdapterManifest {
  readonly schema: 'b8p.adapter.v1';
  readonly digest: string;
  readonly casPath: string;
  readonly adapterName: string;
  readonly baseModel: string;
  readonly recipeId: UnslothRecipeId;
  readonly quant: GgufQuantId;
  readonly bitsPerWeight: number;
  readonly estimatedSizeMb: number;
  readonly dynamicQuantization2_0: boolean;
  readonly preservedHighPrecisionLayers: readonly string[];
  readonly targetModules: readonly string[];
  readonly lora: {
    readonly r: number;
    readonly alpha: number;
    readonly dropout: number;
    readonly learningRate: number;
    readonly useGradientCheckpointing: 'unsloth' | boolean;
    readonly sequencePacking: boolean;
  };
  readonly embedding?: {
    readonly strategy: EmbedAdapterStrategy;
    readonly matryoshkaDims: readonly number[];
  };
  readonly serializedJson: string;
}

export interface LlamaCppExportSpec {
  readonly baseF16Path: string;
  readonly quantizedGgufPath: string;
  readonly loraGgufPath: string;
  readonly quantizeArgs: readonly string[];
  readonly serverArgs: readonly string[];
  readonly quantizeCommand: string;
  readonly serverCommand: string;
}

export interface GpuWorkerDispatchSpec {
  readonly targetMesh: 'CloudDeviceMesh';
  readonly nodeId: string;
  readonly nodeTier: 'cuda-24gb' | 'cuda-48gb' | 'cuda-80gb' | 'cuda-multi-gpu';
  readonly estimatedVramGb: number;
  readonly pythonScript: string;
}

export interface UnslothExportPlan {
  readonly recipe: UnslothRecipeSpec;
  readonly quantSpec: GgufQuantSpec;
  readonly estimatedOutputSizeMb: number;
  readonly estimatedTrainingVramGb: number;
  readonly b8pCasManifest: B8pCasAdapterManifest;
  readonly ollamaModelfile: string;
  readonly llamaCpp: LlamaCppExportSpec;
  readonly gpuWorkerDispatch: GpuWorkerDispatchSpec | null;
}

function sanitizeSlug(input: string): string {
  return input
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'adapter';
}

function selectCloudMeshTier(vramGb: number): GpuWorkerDispatchSpec['nodeTier'] {
  if (vramGb <= 20) {
    return 'cuda-24gb';
  }
  if (vramGb <= 42) {
    return 'cuda-48gb';
  }
  if (vramGb <= 74) {
    return 'cuda-80gb';
  }
  return 'cuda-multi-gpu';
}

function buildGpuWorkerPythonScript(params: {
  modelId: string;
  adapterName: string;
  recipe: UnslothRecipeSpec;
  datasetPath: string;
  targetModules: readonly string[];
  loraRank: number;
  loraAlpha: number;
  loraDropout: number;
  learningRate: number;
  maxSeqLength: number;
  epochs: number;
  batchSize: number;
  gradAccum: number;
  quant: GgufQuantId;
  nodeId: string;
  embedStrategy?: EmbedAdapterStrategy;
  matryoshkaDims?: readonly number[];
}): string {
  const {
    modelId,
    adapterName,
    recipe,
    datasetPath,
    targetModules,
    loraRank,
    loraAlpha,
    loraDropout,
    learningRate,
    maxSeqLength,
    epochs,
    batchSize,
    gradAccum,
    quant,
    nodeId,
    embedStrategy,
    matryoshkaDims,
  } = params;

  const quantMethodLower = quant.toLowerCase();
  const targetListPy = JSON.stringify([...targetModules]);
  const loadIn4BitPy = recipe.loadIn4Bit ? 'True' : 'False';
  const packingPy = recipe.sequencePacking ? 'True' : 'False';
  const gcPy =
    recipe.useGradientCheckpointing === 'unsloth'
      ? '"unsloth"'
      : recipe.useGradientCheckpointing
        ? 'True'
        : 'False';

  const trainerImport =
    recipe.trainerClass === 'ContrastiveLoRATrainer'
      ? 'from sentence_transformers import SentenceTransformer, losses'
      : `from trl import ${recipe.trainerClass}`;

  const extraComment =
    recipe.id === 'embed-jina-v3-lora'
      ? `# Embedding strategy: ${embedStrategy ?? 'direct-task-finetune'} | Matryoshka dims: ${JSON.stringify(matryoshkaDims ?? [1024, 512, 256, 128, 64])}`
      : `# Dynamic 2.0 Quantization: ${recipe.dynamicQuantization2_0 ? 'enabled' : 'disabled'}`;

  return [
    `# Sovereign Unsloth GPU Worker Dispatch — CloudDeviceMesh node: ${nodeId}`,
    `# Recipe: ${recipe.id} (${recipe.name})`,
    extraComment,
    'from unsloth import FastLanguageModel',
    'from datasets import load_dataset',
    trainerImport,
    'from transformers import TrainingArguments',
    '',
    `MODEL_ID = ${JSON.stringify(modelId)}`,
    `ADAPTER_NAME = ${JSON.stringify(adapterName)}`,
    `DATASET_PATH = ${JSON.stringify(datasetPath)}`,
    `MAX_SEQ_LENGTH = ${maxSeqLength}`,
    '',
    'model, tokenizer = FastLanguageModel.from_pretrained(',
    '    model_name=MODEL_ID,',
    '    max_seq_length=MAX_SEQ_LENGTH,',
    '    dtype=None,',
    `    load_in_4bit=${loadIn4BitPy},`,
    ')',
    '',
    'model = FastLanguageModel.get_peft_model(',
    '    model,',
    `    r=${loraRank},`,
    `    target_modules=${targetListPy},`,
    `    lora_alpha=${loraAlpha},`,
    `    lora_dropout=${loraDropout},`,
    '    bias="none",',
    `    use_gradient_checkpointing=${gcPy},`,
    '    random_state=3407,',
    ')',
    '',
    'dataset = load_dataset("json", data_files=DATASET_PATH, split="train")',
    '',
    'training_args = TrainingArguments(',
    `    output_dir=f"./outputs/{ADAPTER_NAME}",`,
    `    per_device_train_batch_size=${batchSize},`,
    `    gradient_accumulation_steps=${gradAccum},`,
    `    num_train_epochs=${epochs},`,
    `    learning_rate=${learningRate},`,
    '    optim="adamw_8bit",',
    '    logging_steps=10,',
    '    bf16=True,',
    ')',
    '',
    recipe.trainerClass === 'ContrastiveLoRATrainer'
      ? `# Execute contrastive Matryoshka triplet training (${embedStrategy ?? 'direct-task-finetune'})`
      : `trainer = ${recipe.trainerClass}(model=model, tokenizer=tokenizer, train_dataset=dataset, args=training_args${recipe.sequencePacking ? `, packing=${packingPy}` : ''})`,
    recipe.trainerClass === 'ContrastiveLoRATrainer'
      ? 'print("Prepared contrastive embedding LoRA model for CloudDeviceMesh execution")'
      : 'trainer.train()',
    '',
    `model.save_pretrained_gguf(f"./exports/{ADAPTER_NAME}", tokenizer, quantization_method=${JSON.stringify(quantMethodLower)})`,
  ].join('\n');
}

export function buildUnslothExportPlan(config: UnslothExportConfig): UnslothExportPlan {
  const recipe = UNSLOTH_RECIPES[config.recipeId];
  if (!recipe) {
    throw new RangeError(`Unknown Unsloth recipe: ${String(config.recipeId)}`);
  }
  const quantSpec = GGUF_EXPORT_QUANTS[config.quant];
  if (!quantSpec) {
    throw new RangeError(`Unknown GGUF quant: ${String(config.quant)}`);
  }

  const slug = sanitizeSlug(config.adapterName);
  const preset = config.targetPreset ?? recipe.defaultTargetPreset;
  const targetModules =
    config.customTargetModules && config.customTargetModules.length > 0
      ? [...config.customTargetModules]
      : [...LORA_TARGET_PRESETS[preset]];

  const loraRank = config.loraRank ?? recipe.defaultRank;
  const loraAlpha = config.loraAlpha ?? recipe.defaultAlpha;
  const loraDropout = config.loraDropout ?? recipe.defaultDropout;
  const learningRate = config.learningRate ?? recipe.defaultLearningRate;
  const maxSeqLength = config.maxSeqLength ?? 2048;
  const epochs = config.epochs ?? 3;
  const batchSize = config.batchSize ?? 2;
  const gradAccum = config.gradientAccumulationSteps ?? 4;
  const contextLength = config.contextLength ?? maxSeqLength;
  const temperature = config.temperature ?? (recipe.category === 'embedding' ? 0 : 0.7);
  const gpuLayers = config.gpuLayers ?? 99;
  const serverPort = config.serverPort ?? 8080;
  const datasetPath = config.datasetPath ?? `./datasets/${slug}.jsonl`;

  const estimatedOutputSizeMb = estimateGgufSizeMb(config.paramBillions, config.quant);
  const estimatedTrainingVramGb = estimateTrainingVramGb(
    config.paramBillions,
    config.recipeId,
    maxSeqLength,
    batchSize,
  );

  const embedInfo =
    config.recipeId === 'embed-jina-v3-lora'
      ? {
          strategy: config.embedAdapterStrategy ?? ('direct-task-finetune' as EmbedAdapterStrategy),
          matryoshkaDims: config.matryoshkaDims ? [...config.matryoshkaDims] : [1024, 512, 256, 128, 64],
        }
      : undefined;

  const casPayload = {
    schema: 'b8p.adapter.v1' as const,
    adapterName: config.adapterName,
    baseModel: config.modelId,
    recipeId: config.recipeId,
    quant: config.quant,
    bitsPerWeight: quantSpec.bitsPerWeight,
    estimatedSizeMb: estimatedOutputSizeMb,
    dynamicQuantization2_0: recipe.dynamicQuantization2_0 || quantSpec.isDynamic2_0,
    preservedHighPrecisionLayers: [...recipe.preservedHighPrecisionLayers],
    targetModules,
    lora: {
      r: loraRank,
      alpha: loraAlpha,
      dropout: loraDropout,
      learningRate,
      useGradientCheckpointing: recipe.useGradientCheckpointing,
      sequencePacking: recipe.sequencePacking,
    },
    ...(embedInfo ? { embedding: embedInfo } : {}),
  };

  const canonicalJson = JSON.stringify(casPayload, null, 2);
  const digest = `sha256:${createHash('sha256').update(canonicalJson).digest('hex')}`;
  const casPath = `.b8g-cas/adapters/${slug}.b8p`;

  const b8pCasManifest: B8pCasAdapterManifest = {
    ...casPayload,
    digest,
    casPath,
    serializedJson: canonicalJson,
  };

  const baseF16Path = `./exports/${slug}-f16.gguf`;
  const quantizedGgufPath = `./exports/${slug}-${config.quant}.gguf`;
  const loraGgufPath = `./exports/${slug}-lora.gguf`;

  const quantizeArgs: readonly string[] = [
    'llama-quantize',
    baseF16Path,
    quantizedGgufPath,
    config.quant,
  ];

  const serverArgs: string[] = [
    'llama-server',
    '-m',
    quantizedGgufPath,
    '--lora',
    loraGgufPath,
    '-c',
    String(contextLength),
    '-ngl',
    String(gpuLayers),
    '--port',
    String(serverPort),
  ];
  if (recipe.category === 'embedding') {
    serverArgs.push('--embedding');
  }

  const ollamaLines: string[] = [
    `FROM ${quantizedGgufPath}`,
    `ADAPTER ${loraGgufPath}`,
    `PARAMETER num_ctx ${contextLength}`,
    `PARAMETER temperature ${temperature}`,
  ];
  if (config.systemPrompt && config.systemPrompt.trim().length > 0) {
    ollamaLines.push(`SYSTEM """${config.systemPrompt.trim()}"""`);
  }

  const shouldGenerateGpuWorker =
    config.includeGpuWorkerScript === true ||
    (config.includeGpuWorkerScript !== false && config.paramBillions >= 7);

  const nodeId = config.cloudMeshNodeId ?? `mesh-${selectCloudMeshTier(estimatedTrainingVramGb)}`;
  const gpuWorkerDispatch: GpuWorkerDispatchSpec | null = shouldGenerateGpuWorker
    ? {
        targetMesh: 'CloudDeviceMesh',
        nodeId,
        nodeTier: selectCloudMeshTier(estimatedTrainingVramGb),
        estimatedVramGb: estimatedTrainingVramGb,
        pythonScript: buildGpuWorkerPythonScript({
          modelId: config.modelId,
          adapterName: config.adapterName,
          recipe,
          datasetPath,
          targetModules,
          loraRank,
          loraAlpha,
          loraDropout,
          learningRate,
          maxSeqLength,
          epochs,
          batchSize,
          gradAccum,
          quant: config.quant,
          nodeId,
          embedStrategy: embedInfo?.strategy,
          matryoshkaDims: embedInfo?.matryoshkaDims,
        }),
      }
    : null;

  return {
    recipe,
    quantSpec,
    estimatedOutputSizeMb,
    estimatedTrainingVramGb,
    b8pCasManifest,
    ollamaModelfile: ollamaLines.join('\n'),
    llamaCpp: {
      baseF16Path,
      quantizedGgufPath,
      loraGgufPath,
      quantizeArgs,
      serverArgs,
      quantizeCommand: quantizeArgs.join(' '),
      serverCommand: serverArgs.join(' '),
    },
    gpuWorkerDispatch,
  };
}

export class UnslothRecipeEngine {
  readonly recipes = UNSLOTH_RECIPES;
  readonly targetPresets = LORA_TARGET_PRESETS;
  readonly ggufQuants = GGUF_EXPORT_QUANTS;

  listRecipes(category?: UnslothRecipeCategory): readonly UnslothRecipeSpec[] {
    const all = Object.values(this.recipes);
    return category ? all.filter((r) => r.category === category) : all;
  }

  getRecipe(id: UnslothRecipeId): UnslothRecipeSpec {
    const recipe = this.recipes[id];
    if (!recipe) {
      throw new RangeError(`Unknown Unsloth recipe: ${String(id)}`);
    }
    return recipe;
  }

  getTargetModules(preset: LoraTargetPresetId): readonly string[] {
    const modules = this.targetPresets[preset];
    if (!modules) {
      throw new RangeError(`Unknown LoRA target preset: ${String(preset)}`);
    }
    return [...modules];
  }

  getQuantSpec(quant: GgufQuantId): GgufQuantSpec {
    const spec = this.ggufQuants[quant];
    if (!spec) {
      throw new RangeError(`Unknown GGUF quant: ${String(quant)}`);
    }
    return spec;
  }

  estimateGgufSizeMb(paramBillions: number, quant: GgufQuantId): number {
    return estimateGgufSizeMb(paramBillions, quant);
  }

  estimateTrainingVramGb(
    paramBillions: number,
    recipeId: UnslothRecipeId,
    maxSeqLength = 2048,
    batchSize = 2,
  ): number {
    return estimateTrainingVramGb(paramBillions, recipeId, maxSeqLength, batchSize);
  }

  validateDataset(format: DatasetFormat, rows: readonly unknown[]): DatasetValidationResult {
    return validateDataset(format, rows);
  }

  buildExportPlan(config: UnslothExportConfig): UnslothExportPlan {
    return buildUnslothExportPlan(config);
  }

  recommendRecipe(criteria: {
    readonly datasetFormat: DatasetFormat;
    readonly availableVramGb?: number;
    readonly paramBillions?: number;
    readonly reasoningRl?: boolean;
    readonly unpairedFeedback?: boolean;
  }): UnslothRecipeSpec {
    if (criteria.datasetFormat === 'contrastive-triplet') {
      return this.recipes['embed-jina-v3-lora'];
    }
    if (criteria.reasoningRl) {
      return this.recipes.grpo;
    }
    if (criteria.datasetFormat === 'preference-dpo') {
      if (criteria.unpairedFeedback) {
        return this.recipes.kto;
      }
      if (criteria.availableVramGb !== undefined && criteria.availableVramGb <= 16) {
        return this.recipes.orpo;
      }
      return this.recipes.dpo;
    }
    const params = criteria.paramBillions ?? 8;
    const vram = criteria.availableVramGb ?? 16;
    const sftVram = this.estimateTrainingVramGb(params, 'sft-lora');
    return vram < sftVram ? this.recipes['qlora-nf4'] : this.recipes['sft-lora'];
  }
}
