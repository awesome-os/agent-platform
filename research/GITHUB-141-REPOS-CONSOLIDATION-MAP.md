# AwesomeOS GitHub Organization Consolidation Map (141-Repo Audit)

> **Audit Scope:** `141` repositories across `github.com/awesome-os/*` (`122` active repositories including the unified `awesome-os/agent-platform` + `19` archived legacy/superseded repositories; `139` local mirrors synced in `e:\awesome-os\github-mirror\awesome-os\`, `116` baseline entries indexed in `github-mirror/manifest.json`).  
> **Consolidation Target:** [`awesome-os/agent-platform`](../README.md) (`@awesome-os/agent-platform`).

---

## 1. Executive Summary

Prior to consolidation, AwesomeOS AI models, MoE routers, inference runtime bridges, embedding/LoRA tooling, Pinokio virtual recipe runners, `x402` agent payment specs, and web dashboards were distributed across standalone repositories and sub-directories of `awesome-os/awesome-os`.

`awesome-os/agent-platform` unifies all sovereign AI, model research, runtime bridges, agent protocols, governance, and browser dashboards into one zero-dependency, `erasableSyntaxOnly: true` monorepo while keeping OS/kernel/browser/compiler repositories cleanly separated in `awesome-os/awesome-os` and sibling device repositories.

| Metric | Count / Value | Notes |
| :--- | :---: | :--- |
| **Total Organization Repositories Audited** | **141** | Across `github.com/awesome-os/*` |
| **Active Repositories** | **122** | Includes `awesome-os/agent-platform` as the new AI hub |
| **Archived Repositories** | **19** | Legacy/superseded `b8g-ghost` / pre-`v11` / superseded experiment repos |
| **Local Mirror Directories Verified** | **139** | Present under `e:\awesome-os\github-mirror\awesome-os\` |
| **Core AI / LLM / Agent / Dashboard Repos Mapped** | **12** | Consolidated directly into `agent-platform/packages/*`, `runtime/*`, `dashboards/*`, `research/*` |

---

## 2. Direct AI, Model, MoE, Inference, Agent, Pinokio, x402 & Dashboard Consolidation Map

Every AI-adjacent repository in the `awesome-os` organization maps into a canonical module inside `awesome-os/agent-platform`:

| Source Repository (`github.com/awesome-os/*`) | Domain | Consolidation Target in `awesome-os/agent-platform` | Role & Status |
| :--- | :--- | :--- | :--- |
| **`awesome-os/models`** | Model Registry & Manifest | [`packages/models-hub/`](../packages/models-hub/index.mts) + [`models/`](../models/README.md) | Unified model registry (`awesome-os-models-manifest-v1`, `B8G_ABI_MAGIC_V3`) & 4 foundational architecture specs (`Inkling 2 Mini`, `B8G Xenon 1B`, `LittleBit/NanoQuant`, `Boundary-Verified MoE`) |
| **`awesome-os/b8g-micro-moe`** | Cache-Resident Micro-MoE (`53.7 KB`) | [`packages/micro-moe/`](../packages/micro-moe/PAPER.md) + [`research/COMPOSABLE-B8P-MICRO-MOE-PAPER.md`](./COMPOSABLE-B8P-MICRO-MOE-PAPER.md) | 5-head INT8 `.b8p` algebra router (`100%` val accuracy, `0.682 ms` latency, `3 ms` online `crystallize()`) |
| **`awesome-os/b8g-app-dedup`** | Micro-MoE Disk/App Classifier (`41.2 KB`) | [`packages/app-dedup-classifier/`](../packages/app-dedup-classifier/path-model.mts) | 3-head INT8 `.b8p` path classifier (`97.62%` held-out accuracy, Bloom OOD gating, active-learning queue) |
| **`awesome-os/colibri`** | Local MoE Inference & Hardware Fit | [`runtime/colibri/`](../runtime/colibri/colibri-v11-bridge.mts) + [`packages/hardware-fit/`](../packages/hardware-fit/hardware-fit.mts) | Colibri `v11` runtime bridge & hardware-fit RAM/VRAM/disk recommender |
| **`awesome-os/llama.cpp`** | Native GGUF Inference Engine | [`runtime/llama-cpp/`](../runtime/llama-cpp/AWESOME-OS-INTEGRATION.md) + [`packages/llm-studio/hub-gguf.mts`](../packages/llm-studio/hub-gguf.mts) | Zero-copy `v11` CAS-backed GGUF v2/v3 parser, `llama-quantize`, `llama-server`, and Ollama `Modelfile` integration |
| **`awesome-os/vllm`** | High-Throughput Serving Engine | [`runtime/vllm/`](../runtime/vllm/AWESOME-OS-INTEGRATION.md) | `v11-max-lock.json` pinned GPU/ROCm/Vulkan serving integration for Looped MoE (`Inkling 2 Mini` / `vLLM`) |
| **`awesome-os/pinokio`** | One-Click AI App Recipes | [`packages/pinokio-recipes/`](../packages/pinokio-recipes/src/index.mts) | Zero-Electron Pinokio recipe runner (`install.json` / `start.json`) with Holo `v11` CAS PyPI wheel interception & Sectium surface |
| **`awesome-os/x402-BaaS-mcp`** | Agent Commerce & Settlement | [`packages/protocols/index.mts`](../packages/protocols/index.mts) + [`research/X402-BAAS-A2A-PAYMENTS.md`](./X402-BAAS-A2A-PAYMENTS.md) | HTTP `402` (`x402`) A2A settlement, BaaS/EMI/CaaS provider registry, A2A, A2UI & 6-tool MCP server |
| **`awesome-os/holo-v11`** | Planetary CAS Oversight Telemetry | [`dashboards/oversight.html`](../dashboards/oversight.html) (via [`dashboards/build-dashboards.mts`](../dashboards/build-dashboards.mts)) | Live upstream library & SHA-512 Holo `v11` CAS oversight dashboard |
| **`awesome-os/host-manager`** | Fleet MDM & Device Console | [`dashboards/host-manager.html`](../dashboards/host-manager.html) (via [`dashboards/build-dashboards.mts`](../dashboards/build-dashboards.mts)) | 4-column Fleet Host Manager console embedded into the Unified Control Center |
| **`awesome-os/b8g.app`** | Sovereign Compute Rack Portal | [`packages/llm-studio/cloud-devices.mts`](../packages/llm-studio/cloud-devices.mts) & [`dashboards/index.html`](../dashboards/index.html) | `b8g.dev` dark-glass aesthetic & remote compute node / lease integration |
| **`awesome-os/awesome-os`** (`ai/`, `sections/`, `intersections/`) | Core Platform & Research Modules | [`packages/model-ops/`](../packages/model-ops/index.mts), [`packages/moe-gate/`](../packages/moe-gate/index.mts), [`packages/atlas-kv/`](../packages/atlas-kv/index.mts), [`packages/kalaschnikow-xenon/`](../packages/kalaschnikow-xenon/README.md), [`packages/spinwave-moe/`](../packages/spinwave-moe/index.mts), [`packages/llm-studio/`](../packages/llm-studio/index.mts), [`packages/agent-host/`](../packages/agent-host/server.mts), [`packages/brain-rag/`](../packages/brain-rag/brain-router.mts), [`packages/error-loop-lsp/`](../packages/error-loop-lsp/index.mts), [`packages/schema/`](../packages/schema/index.mts), [`packages/zer0q-substrate/`](../packages/zer0q-substrate/index.mts), [`packages/governance/`](../packages/governance/index.mts), [`packages/observability-eval/`](../packages/observability-eval/index.mts) | Full AI/Agent/MoE stack consolidated into standalone `@awesome-os/agent-platform` packages |

---

## 3. Non-AI Ecosystem Boundary (Kept in Dedicated Sister Repos)

To prevent `agent-platform` from inheriting multi-gigabyte C++/OS/browser build trees, the remaining active repositories remain cleanly decoupled and interoperate via `B8G_ABI_MAGIC_V3` (`0x33564742`) and Holo `v11` CAS:

| Ecosystem Pillar | Representative Organization Repositories | Boundary Interface to `agent-platform` |
| :--- | :--- | :--- |
| **OS, Kernel & Boot** | `awesome-os`, `t2linux`, `t2linux-baseline`, `t2xnu`, `t2desktop-noctalia`, `bootloader`, `installer`, `deployment`, `docker`, `dockerimages`, `firecracker`, `coreutils`, `posix`, `wayland`, `noctalia`, `siliconforest`, `siliconforest-m1n1` | `B8G_ARENA_V1` (`/dev/b8g-arena`), `AF_VSOCK`, `Host Manager` console |
| **Sovereign Browsers & Display** | `sectium`, `sectium-browser`, `sectium-display`, `sectium-devtools`, `sectium-electron`, `ladybird`, `blink`, `dawn`, `skia`, `brave`, `electron`, `qt`, `webui`, `wry` | Dual NDJSON pipe (`\\.\pipe\sectium-dev`), `Sectium Shell`, `<b8g-*>` A2UI |
| **IDEs & Editor Bridges** | `code-oss`, `b8g-ide`, `theia`, `stealifylang-vscode`, `blender-vscode`, `godot-vscode`, `obs-vscode`, `unity-vscode`, `unreal-vscode` | Persistent `Agent Host` SSE (`:7498`), `error-loop-lsp` LSP 3.17 diagnostics, `brain-rag` |
| **Compilers, Runtimes & Toolchains** | `holo-compiler`, `holo-ffi`, `holo-pm`, `holo-sdk`, `node`, `bun-neon`, `libv8_monolith`, `lo`, `lo-bench`, `lo-types`, `just`, `just-js`, `typescript`, `ts-morph`, `ts-sema`, `rollup`, `pnpm`, `cargo`, `uv`, `dotnet`, `go-lang`, `powershell`, `wasi-host` | `holo-compiler` native C11/WASI builds (`b8g_model_ops_abi.h`), direct Node 24+ `.mts` execution |
| **Native Libs, Infra & Benchmarks** | `libuv`, `ws`, `pouchdb`, `modern-picomatch`, `ory-polis`, `universal-git`, `gitaly`, `upstream-mirrors`, `benchmarks`, `benchy`, `pgbench`, `techempower`, `awesome-os.github.io`, `b8g-production-rack` | Upstream `v11` CAS pinning (`oversight.html`), public web portal hosting (`b8g.dev/awesome-os/apps/`) |

---

## 4. Archived Repository Disposition (`19` Archived Repos)

During the `141`-repo consolidation sweep, **`19` legacy repositories** were designated archived / superseded by the Holo `v11` CAS (`awesome-os/holo-v11`), `holo-compiler`, and `awesome-os/agent-platform`:

- **Legacy CAS-v1 / Ghost-Era Repositories (`b8g-ghost`, `ghost-100`, `ghost-cpp`, `ghost-jvm`, `ghost-py`):** Superseded by `awesome-os/holo-v11` (`B8G_ABI_MAGIC_V3 = 0x33564742`, SHA-512 `v11/files/<aa>/<sha512>`), `holo-pm`, and `sectium-es` / `sectium-jvm`. Zero `b8g-ghost` or `cas-v1` runtime references remain in `agent-platform`.
- **Superseded Pre-Self-Hosted Bootstrappers & Prototype Repos (14 archived org repos):** Legacy standalone prototypes, early stage-0 bootstrappers, and split pre-monorepo experiments consolidated into `awesome-os/awesome-os` (`repository/*` archival candidates in `B8G-SELF-HOSTED-DEBLOAT-AUDIT.md`) and `awesome-os/agent-platform` (`models/`, `packages/`, `dashboards/`).

---

## 5. Verification Gates

All consolidated packages and dashboards in `awesome-os/agent-platform` are verified continuously via [`.github/workflows/ci.yml`](../.github/workflows/ci.yml):

1. `cc -std=c11 -Wall -Wextra -Werror -fsyntax-only -x c packages/model-ops/b8g_model_ops_abi.h`
2. `npm run build:dashboards` (`node dashboards/build-dashboards.mts`)
3. `npm test` (29 package test suites, 155 passing tests)
4. `npm run tutorials` (`tutorials/01` through `tutorials/05`)
