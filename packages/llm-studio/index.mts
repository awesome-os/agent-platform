/**
 * @file index.mts
 * @description Canonical 7-Tuple Algebraic Intersection (`B8gLlmStudioSection`) unifying:
 *   - LM Studio + Ollama + Hugging Face GGUF (`UnifiedModelHub` / `llama.cpp`)
 *   - Jina-v3 / ModernBERT / lumiBERT Task-LoRA & Matryoshka Embeddings (`LoraStudioEngine`)
 *   - Unsloth Dynamic 2.0 / QLoRA / SFT / DPO / GRPO / ORPO / KTO Recipes (`UnslothRecipeEngine`)
 *   - Google Colab GPU Remote Host Node & Universal Notebook Bridge (`ColabNodeBridge` + `CloudDeviceMesh`)
 *   - Sovereign `globalThis.ai` Local Bridge (`SovereignAiHostBridge`, 0 closed Google DLLs)
 *
 * Algebra:
 *   LlmStudio = ModelHub ^ LoraEngine ^ UnslothRecipes ^ ColabHostMesh ^ GlobalAiBridge
 */
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import {
  defineSection,
  f,
  p,
  s,
  type ExecutionContext,
} from '../schema/index.mts';
import { SovereignAiHostBridge } from './global-ai-shim.mts';
import { CloudDeviceMesh } from './cloud-devices.mts';
import {
  LoraStudioEngine,
  cosineSimilarity,
  estimateUnslothMemory,
  sliceMatryoshka,
  type ContrastivePair,
  type LoraStrategy,
  type QuantizationKind,
  type TaskAdapterId,
  type TrainStepEvent,
} from './lora-engine.mts';
import {
  UnifiedModelHub,
  buildLlamaCppLaunchArgs,
  parseOllamaModelfile,
  searchHuggingFaceGguf,
} from './hub-gguf.mts';
import {
  UNSLOTH_RECIPES,
  buildUnslothExportPlan,
  validateDataset,
  type DatasetFormat,
  type UnslothExportConfig,
} from './unsloth-recipes.mts';
import {
  ColabNodeBridge,
  generateColabNotebookIpynb,
  parseUniversalNotebookCells,
  type ColabGpuTier,
  type GenerateColabNotebookOptions,
} from './colab-node-bridge.mts';

export const LLM_STUDIO_CONTEXT: ExecutionContext = {
  boundary: 'host',
  display: 'window',
  lifetime: 'session',
  transport: 'sab',
  capabilities: { fs: true, net: true, proc: true },
  locality: 'local-host',
  persona: 'creator',
};

export const B8gLlmStudioSection = defineSection({
  id: 'b8g.llm.studio',
  version: '1.0.0',
  kind: 'ui-surface',
  shape: {
    surface: 'canvas-2d',
    liveness: 'stream',
    htmlTagged: true,
    actions: ['chat', 'embed', 'lora-train', 'lora-merge', 'hub-search', 'colab-dispatch'],
  },
  description:
    'Unified Sovereign LLM, Embeddings & Unsloth/Jina-v3 LoRA Studio (OpenAI /v1/* + Ollama /api/* + Hugging Face GGUF + llama.cpp + Google Colab GPU Host Nodes).',
  context: LLM_STUDIO_CONTEXT,
  storage: s.sabRing({ capacityBytes: 256 * 1024 }),
  fields: {
    activeModel: f.string({ default: 'lumiBERT-local' }),
    activeTaskAdapter: f.string({ default: 'retrieval.query' }),
    colabNodesOnline: f.int32({ default: 0 }),
  },
  protocols: {
    chat: p.input('rpc', 'b8g.llm.chat'),
    embeddings: p.input('rpc', 'b8g.llm.embeddings'),
    trainStream: p.output('stream', 'b8g.llm.trainStream'),
  },
});

export interface SovereignLlmStudioOptions {
  casModelsDir?: string;
  deviceMesh?: CloudDeviceMesh;
}

export class SovereignLlmStudio {
  readonly lora: LoraStudioEngine;
  readonly hub: UnifiedModelHub;
  readonly ai: SovereignAiHostBridge;
  readonly mesh: CloudDeviceMesh;
  readonly colab: ColabNodeBridge;
  readonly casModelsDir: string | null;

  constructor(opts: SovereignLlmStudioOptions = {}) {
    this.casModelsDir = opts.casModelsDir ? resolve(opts.casModelsDir) : null;
    this.lora = new LoraStudioEngine({ dim: 64, defaultRank: 8, defaultAlpha: 16 });
    this.hub = new UnifiedModelHub({
      localDirs: this.casModelsDir ? [this.casModelsDir] : [],
    });
    this.ai = new SovereignAiHostBridge();
    this.mesh = opts.deviceMesh ?? new CloudDeviceMesh({ autoRegisterLocalHost: true });
    this.colab = new ColabNodeBridge({ mesh: this.mesh });
  }

  /**
   * OpenAI `/v1/embeddings` & Ollama `/api/embeddings` compatible encoder using
   * the live trained `LoraStudioEngine` task adapter + Matryoshka dimension slicing.
   */
  embedTexts(opts: {
    inputs: string[];
    model?: string;
    task?: TaskAdapterId;
    dimensions?: number;
  }): {
    object: 'list';
    model: string;
    task: TaskAdapterId;
    dimensions: number;
    data: Array<{ object: 'embedding'; index: number; embedding: number[] }>;
    pairwiseCosine?: number;
  } {
    const task: TaskAdapterId = opts.task ?? 'retrieval.query';
    const dim = Math.max(8, Math.min(64, opts.dimensions ?? 64));
    const vecs = opts.inputs.map((txt) => sliceMatryoshka(this.lora.embed(txt, task), dim));
    const data = vecs.map((v, index) => ({
      object: 'embedding' as const,
      index,
      embedding: Array.from(v, (n) => Number(n.toFixed(6))),
    }));
    const pairwiseCosine =
      vecs.length >= 2 ? Number(cosineSimilarity(vecs[0]!, vecs[1]!).toFixed(4)) : undefined;

    return {
      object: 'list',
      model: opts.model ?? 'jina-embeddings-v3-gguf',
      task,
      dimensions: dim,
      data,
      pairwiseCosine,
    };
  }

  /**
   * Trains a Task-LoRA adapter via pure Web Stream (`trainStream`) and returns
   * the full loss trajectory + live VRAM estimate.
   */
  async trainLora(opts: {
    taskId: TaskAdapterId;
    strategy: LoraStrategy;
    pairs: ContrastivePair[];
    steps?: number;
    lr?: number;
    rank?: number;
    quant?: QuantizationKind;
    paramsBillion?: number;
  }): Promise<{
    taskId: TaskAdapterId;
    strategy: LoraStrategy;
    steps: TrainStepEvent[];
    adapterSha256: string;
    memoryEstimate: ReturnType<typeof estimateUnslothMemory>;
  }> {
    const stepsCount = Math.max(2, Math.min(60, opts.steps ?? 16));
    const lr = opts.lr ?? 0.08;
    const rank = opts.rank ?? 8;

    if (opts.strategy === 'merge-then-stack') {
      this.lora.applyStrategy(opts.taskId, 'merge-then-stack', { rank });
    } else {
      this.lora.createOrResetAdapter(opts.taskId, { rank, strategy: 'direct-task-finetune' });
    }

    const stream = this.lora.trainStream(opts.taskId, opts.pairs, { steps: stepsCount, lr });
    const reader = stream.getReader();
    const events: TrainStepEvent[] = [];
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        events.push(value);
      }
    } finally {
      reader.releaseLock();
    }

    const adapter = this.lora.getAdapter(opts.taskId);
    const memoryEstimate = estimateUnslothMemory(
      opts.paramsBillion ?? 0.57,
      rank,
      opts.quant ?? 'nf4',
      Math.max(1, opts.pairs.length),
    );

    return {
      taskId: opts.taskId,
      strategy: opts.strategy,
      steps: events,
      adapterSha256: adapter.sha256,
      memoryEstimate,
    };
  }

  /**
   * Merges `W' = W + (alpha/r) * B * A` and optionally persists a Merkle-auditable
   * `.b8p` adapter manifest into `.b8g-cas/models/` so `CasModuleRegistry` picks it up!
   */
  mergeAndPersistAdapter(taskId: TaskAdapterId): {
    taskId: TaskAdapterId;
    sha256: string;
    bytes: number;
    persistedPath?: string;
  } {
    const merged = this.lora.mergeAdapter(taskId);
    let persistedPath: string | undefined;
    if (this.casModelsDir) {
      if (!existsSync(this.casModelsDir)) mkdirSync(this.casModelsDir, { recursive: true });
      const file = join(this.casModelsDir, `lora-${taskId.replace(/\./g, '-')}-${merged.sha256.slice(0, 12)}.json`);
      writeFileSync(
        file,
        JSON.stringify(
          {
            schema: 'b8g-lora-merged-v1',
            taskId,
            sha256: merged.sha256,
            bytes: merged.bytes,
            mergedAt: new Date().toISOString(),
          },
          null,
          2,
        ),
        'utf8',
      );
      persistedPath = file;
    }
    return { taskId, sha256: merged.sha256, bytes: merged.bytes, persistedPath };
  }

  /**
   * Registers a Google Colab GPU Host Node into `CloudDeviceMesh` and emits a
   * runnable nbformat v4.5 `.ipynb` Unsloth/Jina-v3 intersection notebook.
   */
  spawnColabGpuIntersection(opts: {
    tenantId?: string;
    label?: string;
    gpuTier?: ColabGpuTier;
    notebookOpts: GenerateColabNotebookOptions;
  }) {
    const device = this.colab.registerNode(
      {
        label: opts.label ?? 'Colab-GPU-Node',
        gpuTier: opts.gpuTier ?? 'L4-24GB',
      },
      opts.tenantId ?? 'root-host',
    );
    const ipynb = generateColabNotebookIpynb({
      ...opts.notebookOpts,
      deviceId: device.deviceId,
    });
    return { node: { device }, ipynb };
  }
}

export {
  LoraStudioEngine,
  UnifiedModelHub,
  ColabNodeBridge,
  UNSLOTH_RECIPES,
  buildLlamaCppLaunchArgs,
  buildUnslothExportPlan,
  generateColabNotebookIpynb,
  parseOllamaModelfile,
  parseUniversalNotebookCells,
  searchHuggingFaceGguf,
  validateDataset,
  type DatasetFormat,
  type UnslothExportConfig,
};
