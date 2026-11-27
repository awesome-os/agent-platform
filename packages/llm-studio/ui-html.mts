/**
 * @file ui-html.mts
 * @description Sovereign LM Studio + Ollama + Hugging Face + Unsloth/Jina-v3 LoRA Studio UI.
 * Dark-glass 5-tab interface with 3px animated Liveness Progress Bar, elapsed-ms spinner pill,
 * SSE Chat, Matryoshka Embeddings Lab, Canvas 2D LoRA Loss Curve, Model Hub, and Unified API Explorer.
 */

const html = (strings: TemplateStringsArray, ...values: unknown[]): string =>
  strings.reduce((acc, s, i) => acc + s + (i < values.length ? String(values[i]) : ''), '');

export function renderLlmStudioHtml(): string {
  return html`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>B8G Sovereign LLM &amp; LoRA Studio</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg%20xmlns=%22http://www.w3.org/2000/svg%22%20viewBox=%220%200%20256%20256%22%20width=%22256%22%20height=%22256%22%20role=%22img%22%20aria-label=%22Sectium%20LLM%22%3E%3Cdefs%3E%3CradialGradient%20id=%22b8g-sectium-llm-bg%22%20cx=%2250%25%22%20cy=%2242%25%22%20r=%2260%25%22%3E%3Cstop%20offset=%220%25%22%20stop-color=%22%23172542%22/%3E%3Cstop%20offset=%22100%25%22%20stop-color=%22%23070d1a%22/%3E%3C/radialGradient%3E%3ClinearGradient%20id=%22b8g-sectium-llm-acc%22%20x1=%220%25%22%20y1=%220%25%22%20x2=%22100%25%22%20y2=%22100%25%22%3E%3Cstop%20offset=%220%25%22%20stop-color=%22%23a855f7%22/%3E%3Cstop%20offset=%22100%25%22%20stop-color=%22%23c084fc%22/%3E%3C/linearGradient%3E%3ClinearGradient%20id=%22b8g-sectium-llm-sec%22%20x1=%220%25%22%20y1=%22100%25%22%20x2=%22100%25%22%20y2=%220%25%22%3E%3Cstop%20offset=%220%25%22%20stop-color=%22%23c084fc%22/%3E%3Cstop%20offset=%22100%25%22%20stop-color=%22%23ffffff%22%20stop-opacity=%220.9%22/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect%20x=%228%22%20y=%228%22%20width=%22240%22%20height=%22240%22%20rx=%2256%22%20fill=%22url(%23b8g-sectium-llm-bg)%22/%3E%3Crect%20x=%229%22%20y=%229%22%20width=%22238%22%20height=%22238%22%20rx=%2255%22%20fill=%22none%22%20stroke=%22%23a855f7%22%20stroke-width=%221.5%22%20stroke-opacity=%220.28%22/%3E%3Ccircle%20cx=%22128%22%20cy=%22128%22%20r=%22106%22%20fill=%22none%22%20stroke=%22%23a855f7%22%20stroke-width=%228%22%20stroke-opacity=%220.92%22/%3E%3Ccircle%20cx=%22128%22%20cy=%22128%22%20r=%2294%22%20fill=%22none%22%20stroke=%22%23c084fc%22%20stroke-width=%222%22%20stroke-opacity=%220.35%22/%3E%3Crect%20x=%2256%22%20y=%2272%22%20width=%22144%22%20height=%22112%22%20rx=%2218%22%20fill=%22%23050b16%22%20stroke=%22%23c084fc%22%20stroke-width=%223%22%20stroke-opacity=%220.7%22/%3E%3Cline%20x1=%2266%22%20y1=%22100%22%20x2=%22190%22%20y2=%22100%22%20stroke=%22%23334155%22%20stroke-width=%221.5%22%20stroke-opacity=%220.45%22/%3E%3Cline%20x1=%2266%22%20y1=%22128%22%20x2=%22190%22%20y2=%22128%22%20stroke=%22%23334155%22%20stroke-width=%221.5%22%20stroke-opacity=%220.65%22/%3E%3Cline%20x1=%2266%22%20y1=%22156%22%20x2=%22190%22%20y2=%22156%22%20stroke=%22%23334155%22%20stroke-width=%221.5%22%20stroke-opacity=%220.45%22/%3E%3Cline%20x1=%2292%22%20y1=%2280%22%20x2=%2292%22%20y2=%22176%22%20stroke=%22%23334155%22%20stroke-width=%221.5%22%20stroke-opacity=%220.4%22/%3E%3Cline%20x1=%22128%22%20y1=%2280%22%20x2=%22128%22%20y2=%22176%22%20stroke=%22%23334155%22%20stroke-width=%221.5%22%20stroke-opacity=%220.5%22/%3E%3Cline%20x1=%22164%22%20y1=%2280%22%20x2=%22164%22%20y2=%22176%22%20stroke=%22%23334155%22%20stroke-width=%221.5%22%20stroke-opacity=%220.4%22/%3E%3Cpath%20d=%22M64%20132%20L82%20132%20L94%20110%20L106%20156%20L120%2086%20L136%20168%20L150%20104%20L162%20138%20L174%20124%20L192%20132%22%20fill=%22none%22%20stroke=%22%23a855f7%22%20stroke-width=%2211%22%20stroke-linecap=%22round%22%20stroke-linejoin=%22round%22%20stroke-opacity=%220.28%22/%3E%3Cpath%20d=%22M64%20132%20L82%20132%20L94%20110%20L106%20156%20L120%2086%20L136%20168%20L150%20104%20L162%20138%20L174%20124%20L192%20132%22%20fill=%22none%22%20stroke=%22url(%23b8g-sectium-llm-acc)%22%20stroke-width=%225.5%22%20stroke-linecap=%22round%22%20stroke-linejoin=%22round%22/%3E%3Ccircle%20cx=%22174%22%20cy=%22124%22%20r=%224.5%22%20fill=%22%23ffffff%22/%3E%3Ccircle%20cx=%2272%22%20cy=%2286%22%20r=%224%22%20fill=%22%23a855f7%22/%3E%3C/svg%3E" />
<style>
  :root {
    --bg: #070b14;
    --bg-glow: radial-gradient(circle at 12% 0%, rgba(56, 189, 248, 0.14), transparent 36%),
               radial-gradient(circle at 88% 8%, rgba(167, 139, 250, 0.14), transparent 40%),
               #070b14;
    --glass: rgba(15, 23, 42, 0.72);
    --glass-2: rgba(22, 33, 56, 0.78);
    --glass-solid: #0f172a;
    --border: rgba(148, 163, 184, 0.18);
    --border-hi: rgba(167, 139, 250, 0.45);
    --accent: #a78bfa;
    --accent-2: #38bdf8;
    --ok: #34d399;
    --warn: #fbbf24;
    --danger: #fb7185;
    --text: #e8eef9;
    --muted: #94a3b8;
    --mono: ui-monospace, 'Cascadia Code', Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    height: 100%;
    background: var(--bg-glow);
    color: var(--text);
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    font-size: 13px;
    overflow: hidden;
  }

  /* Fixed top 3px animated Liveness Progress Bar */
  .liveness-bar {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, #38bdf8, #a78bfa, #34d399, #38bdf8);
    background-size: 250% 100%;
    opacity: 0;
    pointer-events: none;
    z-index: 9999;
    transition: opacity 0.16s ease;
  }
  .liveness-bar.active {
    opacity: 1;
    animation: livenessSlide 1.05s linear infinite;
  }
  @keyframes livenessSlide {
    0% { background-position: 0% 50%; }
    100% { background-position: 250% 50%; }
  }

  /* Elapsed-ms Spinner Pill */
  .liveness-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(56, 189, 248, 0.45);
    color: #bae6fd;
    font-family: var(--mono);
    font-size: 11px;
    opacity: 0;
    transform: translateY(-2px);
    pointer-events: none;
    transition: opacity 0.16s ease, transform 0.16s ease;
    white-space: nowrap;
  }
  .liveness-pill.active {
    opacity: 1;
    transform: translateY(0);
  }
  .spinner {
    width: 12px;
    height: 12px;
    border: 2px solid rgba(56, 189, 248, 0.25);
    border-top-color: #38bdf8;
    border-radius: 50%;
    animation: spin 0.65s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* App Shell */
  .app {
    display: grid;
    grid-template-rows: 54px 1fr;
    height: 100vh;
  }
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 0 16px;
    background: rgba(11, 17, 32, 0.86);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--border);
  }
  .brand-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    white-space: nowrap;
  }
  .brand-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: linear-gradient(135deg, #38bdf8, #a78bfa);
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.7);
  }
  .brand {
    font-weight: 800;
    letter-spacing: 0.4px;
    font-size: 14px;
    background: linear-gradient(90deg, #e0f2fe, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 7px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.3px;
    background: rgba(167, 139, 250, 0.16);
    color: #ddd6fe;
    border: 1px solid rgba(167, 139, 250, 0.3);
  }
  .badge.ok { background: rgba(52, 211, 153, 0.14); color: #6ee7b7; border-color: rgba(52, 211, 153, 0.35); }
  .badge.sky { background: rgba(56, 189, 248, 0.14); color: #7dd3fc; border-color: rgba(56, 189, 248, 0.35); }
  .badge.warn { background: rgba(251, 191, 36, 0.14); color: #fde68a; border-color: rgba(251, 191, 36, 0.35); }

  /* Tabs */
  .tabs {
    display: flex;
    align-items: center;
    gap: 6px;
    overflow-x: auto;
  }
  .tab-btn {
    background: transparent;
    color: var(--muted);
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 6px 12px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    transition: all 0.15s ease;
    white-space: nowrap;
  }
  .tab-btn:hover {
    color: var(--text);
    background: rgba(255, 255, 255, 0.04);
    border-color: var(--border);
  }
  .tab-btn.active {
    color: #fff;
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.28), rgba(14, 165, 233, 0.24));
    border-color: var(--border-hi);
    box-shadow: 0 0 16px rgba(139, 92, 246, 0.18);
  }

  /* Main Stage */
  .main {
    min-height: 0;
    position: relative;
    overflow: hidden;
  }
  .tab-pane {
    display: none;
    height: 100%;
    padding: 14px 16px;
    overflow: auto;
  }
  .tab-pane.active {
    display: block;
  }

  /* Layout & Glass Panels */
  .grid-chat { display: grid; grid-template-columns: 310px 1fr; gap: 14px; height: 100%; min-height: 520px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .grid-lora { display: grid; grid-template-columns: 400px 1fr; gap: 14px; min-height: 540px; }
  .grid-hub { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 14px; }

  @media (max-width: 980px) {
    .grid-chat, .grid-2, .grid-lora, .grid-hub { grid-template-columns: 1fr; }
  }

  .card {
    background: var(--glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-height: 0;
  }
  .card-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    font-weight: 700;
    font-size: 13px;
    color: #f1f5f9;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12);
  }

  /* Form Controls */
  label.field {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 11px;
    color: var(--muted);
    font-weight: 600;
  }
  .field-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  input[type="text"], input[type="search"], select, textarea {
    width: 100%;
    background: rgba(6, 10, 20, 0.85);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 7px 10px;
    font-family: inherit;
    font-size: 12px;
    outline: none;
    transition: border-color 0.15s ease;
  }
  textarea { resize: vertical; line-height: 1.45; }
  textarea.mono, pre.code-box, code {
    font-family: var(--mono);
    font-size: 11.5px;
  }
  input:focus, select:focus, textarea:focus {
    border-color: var(--accent-2);
  }
  input[type="range"] {
    width: 100%;
    accent-color: #38bdf8;
    cursor: pointer;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    background: var(--glass-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 7px 12px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    transition: all 0.14s ease;
    white-space: nowrap;
  }
  .btn:hover { border-color: var(--accent); background: rgba(30, 41, 69, 0.9); }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn.primary {
    background: linear-gradient(135deg, #8b5cf6, #0ea5e9);
    border: none;
    color: #fff;
    font-weight: 700;
  }
  .btn.primary:hover { filter: brightness(1.08); }
  .btn.ok {
    background: linear-gradient(135deg, #059669, #0284c7);
    border: none;
    color: #fff;
    font-weight: 700;
  }
  .btn.danger {
    border-color: rgba(251, 113, 133, 0.45);
    color: #fda4af;
  }
  .btn.sm { padding: 4px 9px; font-size: 11px; border-radius: 6px; }

  .seg-group {
    display: flex;
    gap: 4px;
    background: rgba(6, 10, 20, 0.8);
    padding: 3px;
    border-radius: 8px;
    border: 1px solid var(--border);
  }
  .seg-btn {
    flex: 1;
    background: transparent;
    color: var(--muted);
    border: none;
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
  }
  .seg-btn.active {
    background: rgba(139, 92, 246, 0.3);
    color: #fff;
    box-shadow: inset 0 0 0 1px rgba(167, 139, 250, 0.45);
  }

  pre.code-box {
    margin: 0;
    background: rgba(5, 8, 16, 0.92);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 9px 11px;
    color: #cbd5e1;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.45;
  }

  /* Chat Transcript */
  .chat-box {
    flex: 1;
    min-height: 220px;
    overflow: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 8px 4px;
  }
  .msg {
    max-width: 86%;
    padding: 10px 12px;
    border-radius: 10px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg.user {
    align-self: flex-end;
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.28), rgba(14, 165, 233, 0.24));
    border: 1px solid rgba(167, 139, 250, 0.4);
  }
  .msg.assistant {
    align-self: flex-start;
    background: rgba(15, 23, 42, 0.88);
    border: 1px solid var(--border);
  }
  .msg-role {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .chat-input-row {
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }

  /* Similarity Meter */
  .sim-track {
    height: 14px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    overflow: hidden;
    border: 1px solid var(--border);
  }
  .sim-fill {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #fb7185, #fbbf24, #34d399, #38bdf8);
    transition: width 0.25s ease;
  }
  .vec-bars {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 44px;
    padding: 4px;
    background: rgba(5, 8, 16, 0.85);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .vec-bar {
    flex: 1;
    min-width: 2px;
    border-radius: 2px 2px 0 0;
  }

  /* Telemetry KPIs */
  .kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }
  .kpi {
    background: rgba(6, 10, 20, 0.78);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
  }
  .kpi-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.35px; }
  .kpi-val { font-family: var(--mono); font-size: 15px; font-weight: 700; margin-top: 2px; color: #f8fafc; }

  canvas#lossCanvas {
    width: 100%;
    height: 230px;
    background: rgba(5, 8, 16, 0.9);
    border: 1px solid var(--border);
    border-radius: 10px;
    display: block;
  }

  /* Model Hub Cards */
  .model-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 360px;
    overflow: auto;
  }
  .model-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 9px 11px;
    background: rgba(6, 10, 20, 0.78);
    border: 1px solid var(--border);
    border-radius: 9px;
  }
  .model-item:hover { border-color: var(--accent-2); }
  .model-name { font-weight: 700; color: #f1f5f9; font-family: var(--mono); font-size: 12px; }
  .model-meta { font-size: 11px; color: var(--muted); margin-top: 2px; }

  /* Endpoint Table */
  .ep-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    font-size: 12px;
  }
  .ep-table th, .ep-table td {
    text-align: left;
    padding: 7px 8px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12);
    overflow-wrap: anywhere;
  }
  .ep-table th { color: var(--muted); font-size: 11px; text-transform: uppercase; }
  .method-pill {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 700;
  }
  .method-get { background: rgba(52, 211, 153, 0.16); color: #6ee7b7; }
  .method-post { background: rgba(56, 189, 248, 0.16); color: #7dd3fc; }
</style>
</head>
<body>
<div class="liveness-bar" id="livenessBar"></div>

<div class="app">
  <header class="topbar">
    <div class="brand-wrap">
      <span class="brand-dot"></span>
      <span class="brand">B8G LLM &amp; LoRA STUDIO</span>
      <span class="badge">SOVEREIGN</span>
    </div>

    <nav class="tabs" role="tablist">
      <button class="tab-btn active" data-tab="chat">1. Chat (llama.cpp)</button>
      <button class="tab-btn" data-tab="embed">2. Embeddings &amp; Matryoshka</button>
      <button class="tab-btn" data-tab="lora">3. LoRA &amp; Unsloth Fine-Tuner</button>
      <button class="tab-btn" data-tab="hub">4. Model Hub (HF / Ollama)</button>
      <button class="tab-btn" data-tab="api">5. Unified API Server</button>
    </nav>

    <div class="liveness-pill" id="livenessPill">
      <span class="spinner"></span>
      <span id="livenessText">Working...</span>
      <span id="livenessMs">0 ms</span>
    </div>
  </header>

  <main class="main">
    <!-- TAB 1: CHAT -->
    <section class="tab-pane active" id="pane-chat">
      <div class="grid-chat">
        <div class="card">
          <div class="card-title">
            <span>llama.cpp Runtime Config</span>
            <span class="badge sky">/v1/chat/completions</span>
          </div>

          <label class="field">
            <span>Active Model</span>
            <select id="chatModel">
              <option value="qwen2.5-coder-7b-q4_k_m">qwen2.5-coder-7b-q4_k_m</option>
              <option value="llama-3.2-3b-instruct-q4_k_m">llama-3.2-3b-instruct-q4_k_m</option>
              <option value="b8g-moe-router">b8g-moe-router</option>
              <option value="jina-embeddings-v3-gguf">jina-embeddings-v3-gguf</option>
              <option value="lumiBERT-local">lumiBERT-local</option>
            </select>
          </label>

          <label class="field">
            <span>System Prompt</span>
            <textarea id="chatSystem" rows="3">You are B8G Sovereign Assistant running locally via llama.cpp with OpenAI &amp; Ollama compatibility.</textarea>
          </label>

          <label class="field">
            <div class="field-row">
              <span>Temperature</span>
              <span id="valTemp" class="badge">0.70</span>
            </div>
            <input type="range" id="slTemp" min="0" max="2" step="0.05" value="0.7" />
          </label>

          <label class="field">
            <div class="field-row">
              <span>Context Window (-c)</span>
              <span id="valCtx" class="badge">8192</span>
            </div>
            <input type="range" id="slCtx" min="2048" max="32768" step="1024" value="8192" />
          </label>

          <label class="field">
            <div class="field-row">
              <span>GPU Layers (-ngl)</span>
              <span id="valNgl" class="badge">99</span>
            </div>
            <input type="range" id="slNgl" min="0" max="99" step="1" value="99" />
          </label>

          <div class="field-row" style="margin-top:4px;">
            <span style="font-size:11px;color:var(--muted);font-weight:600;">Live llama-server CLI Preview</span>
            <button class="btn sm" id="btnCopyLlama">Copy CLI</button>
          </div>
          <pre class="code-box" id="llamaCmdPreview"></pre>
        </div>

        <div class="card">
          <div class="card-title">
            <span>Streaming SSE Chat Transcript</span>
            <div style="display:flex;gap:6px;">
              <button class="btn sm danger" id="btnStopChat" disabled>Stop</button>
              <button class="btn sm" id="btnClearChat">Clear</button>
            </div>
          </div>

          <div class="chat-box" id="chatTranscript">
            <div class="msg assistant">
              <div class="msg-role">system</div>
              Ready. Send a prompt to stream tokens from <code>POST /v1/chat/completions</code> (SSE).
            </div>
          </div>

          <div class="chat-input-row">
            <textarea id="chatPrompt" rows="2" placeholder="Ask anything... (Ctrl+Enter to send)"></textarea>
            <button class="btn primary" id="btnSendChat" style="height:48px;min-width:96px;">Send SSE</button>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 2: EMBEDDINGS & MATRYOSHKA LAB -->
    <section class="tab-pane" id="pane-embed">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>Embeddings &amp; Matryoshka Inputs</span>
            <span class="badge sky">/v1/embeddings</span>
          </div>

          <div class="grid-2">
            <label class="field">
              <span>Task Adapter (Jina-v3 / lumiBERT)</span>
              <select id="embedAdapter">
                <option value="retrieval.query">retrieval.query</option>
                <option value="retrieval.passage">retrieval.passage</option>
                <option value="text-matching" selected>text-matching</option>
                <option value="error-repair">error-repair</option>
                <option value="code-search">code-search</option>
              </select>
            </label>

            <label class="field">
              <span>Matryoshka Dimension</span>
              <div class="seg-group" id="dimGroup">
                <button class="seg-btn active" data-dim="64">64d</button>
                <button class="seg-btn" data-dim="32">32d</button>
                <button class="seg-btn" data-dim="16">16d</button>
              </div>
            </label>
          </div>

          <label class="field">
            <span>Text A (Anchor / Query)</span>
            <textarea id="embedTextA" rows="4">TypeError: Cannot read properties of undefined (reading 'slice') in Matryoshka vector truncator</textarea>
          </label>

          <label class="field">
            <span>Text B (Candidate / Passage)</span>
            <textarea id="embedTextB" rows="4">Guard undefined Float32Array before calling sliceMatryoshka and renormalizing L2 unit length</textarea>
          </label>

          <div style="display:flex;gap:8px;align-items:center;">
            <button class="btn primary" id="btnComputeEmbed">Compute Embeddings &amp; Cosine Similarity</button>
            <span id="embedStatus" style="font-size:11px;color:var(--muted);"></span>
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <span>Cosine Similarity &amp; Vector Preview</span>
            <span id="embedDimBadge" class="badge ok">64 dims</span>
          </div>

          <div class="kpi-row" style="grid-template-columns: repeat(3, 1fr);">
            <div class="kpi">
              <div class="kpi-label">Cosine Similarity</div>
              <div class="kpi-val" id="simScore">--</div>
            </div>
            <div class="kpi">
              <div class="kpi-label">Vector A L2 Norm</div>
              <div class="kpi-val" id="normA">--</div>
            </div>
            <div class="kpi">
              <div class="kpi-label">Vector B L2 Norm</div>
              <div class="kpi-val" id="normB">--</div>
            </div>
          </div>

          <div>
            <div class="field-row" style="margin-bottom:4px;">
              <span style="font-size:11px;color:var(--muted);">Semantic Alignment</span>
              <span id="simPctLabel" class="badge">0%</span>
            </div>
            <div class="sim-track"><div class="sim-fill" id="simFill"></div></div>
          </div>

          <label class="field">
            <span>Vector A Sparkline</span>
            <div class="vec-bars" id="vecBarsA"></div>
          </label>

          <label class="field">
            <span>Vector B Sparkline</span>
            <div class="vec-bars" id="vecBarsB"></div>
          </label>

          <pre class="code-box" id="vecPreview" style="max-height:140px;">Run Compute Embeddings to inspect normalized Matryoshka vectors.</pre>
        </div>
      </div>
    </section>

    <!-- TAB 3: LORA & UNSLOTH FINE-TUNER -->
    <section class="tab-pane" id="pane-lora">
      <div class="grid-lora">
        <div class="card">
          <div class="card-title">
            <span>LoRA &amp; Unsloth Configuration</span>
            <span class="badge">/api/studio/lora/*</span>
          </div>

          <label class="field">
            <span>Jina-v3 Adaptation Strategy</span>
            <div class="seg-group" id="loraStrategyGroup">
              <button class="seg-btn active" data-strategy="direct-task-finetune">direct-task-finetune</button>
              <button class="seg-btn" data-strategy="merge-then-stack">merge-then-stack</button>
            </div>
          </label>

          <div class="grid-2">
            <label class="field">
              <span>Unsloth Recipe</span>
              <select id="loraRecipe">
                <option value="Contrastive-Embed">Contrastive-Embed</option>
                <option value="SFT-LoRA">SFT-LoRA</option>
                <option value="DPO">DPO</option>
                <option value="GRPO">GRPO</option>
              </select>
            </label>

            <label class="field">
              <span>Task Adapter Target</span>
              <select id="loraTask">
                <option value="error-repair">error-repair</option>
                <option value="code-search">code-search</option>
                <option value="retrieval.query">retrieval.query</option>
                <option value="retrieval.passage">retrieval.passage</option>
                <option value="text-matching">text-matching</option>
              </select>
            </label>
          </div>

          <label class="field">
            <span>Quantization Mode (Live VRAM Estimator)</span>
            <div class="seg-group" id="loraQuantGroup">
              <button class="seg-btn active" data-quant="nf4">NF4 QLoRA</button>
              <button class="seg-btn" data-quant="q8">Q8</button>
              <button class="seg-btn" data-quant="fp16">FP16</button>
            </div>
          </label>

          <div class="kpi-row">
            <div class="kpi">
              <div class="kpi-label">Base MB</div>
              <div class="kpi-val" id="vramBase">4200</div>
            </div>
            <div class="kpi">
              <div class="kpi-label">LoRA+Opt</div>
              <div class="kpi-val" id="vramLora">420</div>
            </div>
            <div class="kpi">
              <div class="kpi-label">Total MB</div>
              <div class="kpi-val" id="vramTotal">4620</div>
            </div>
            <div class="kpi">
              <div class="kpi-label">vs FP16</div>
              <div class="kpi-val" id="vramSave" style="color:var(--ok);">-71%</div>
            </div>
          </div>

          <label class="field">
            <span>Contrastive Training Pairs (Anchor / Positive / Negative JSON)</span>
            <textarea id="loraPairs" class="mono" rows="8">[
  {
    "anchor": "Fix TypeScript erasableSyntaxOnly enum error",
    "positive": "Replace enum with as const object union type",
    "negative": "Install webpack babel downlevel plugin"
  },
  {
    "anchor": "Merge LoRA adapter into base weights without latency",
    "positive": "Compute W_prime = W + (alpha / rank) * B * A",
    "negative": "Restart docker daemon and clear browser cache"
  },
  {
    "anchor": "Truncate embedding vector to 16 Matryoshka dimensions",
    "positive": "Slice first 16 floats and apply L2 unit normalization",
    "negative": "Pad string with trailing spaces to 16 bytes"
  }
]</textarea>
          </label>

          <div style="display:flex;gap:8px;flex-wrap:wrap;">
            <button class="btn primary" id="btnTrainLora" style="flex:1;">Train LoRA</button>
            <button class="btn ok" id="btnMergeLora" style="flex:1;">Merge W' = W + B·A to CAS</button>
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <span>Live Training Loss Curve (Canvas 2D)</span>
            <span id="loraStepBadge" class="badge sky">Idle</span>
          </div>

          <canvas id="lossCanvas" width="760" height="260"></canvas>

          <div class="kpi-row">
            <div class="kpi">
              <div class="kpi-label">Current Loss</div>
              <div class="kpi-val" id="kpiLoss">--</div>
            </div>
            <div class="kpi">
              <div class="kpi-label">Pos Cosine</div>
              <div class="kpi-val" id="kpiPosSim" style="color:var(--ok);">--</div>
            </div>
            <div class="kpi">
              <div class="kpi-label">Neg Cosine</div>
              <div class="kpi-val" id="kpiNegSim" style="color:var(--warn);">--</div>
            </div>
            <div class="kpi">
              <div class="kpi-label">Delta Loss</div>
              <div class="kpi-val" id="kpiDelta">--</div>
            </div>
          </div>

          <div class="card-title" style="margin-top:4px;">
            <span>CAS Merge Receipt &amp; Training Log</span>
            <span id="casDigestBadge" class="badge">Unmerged</span>
          </div>
          <pre class="code-box" id="loraLog" style="flex:1;min-height:110px;">Configure strategy/recipe and click Train LoRA to run contrastive Float32 optimization.</pre>
        </div>
      </div>
    </section>

    <!-- TAB 4: MODEL HUB -->
    <section class="tab-pane" id="pane-hub">
      <div class="grid-hub">
        <div class="card">
          <div class="card-title">
            <span>Hugging Face GGUF Search</span>
            <span class="badge sky">/api/studio/hub/search</span>
          </div>
          <div style="display:flex;gap:8px;">
            <input type="search" id="hubQuery" placeholder="Search Hugging Face GGUF (e.g. qwen, llama, jina, mistral)..." value="qwen" />
            <button class="btn primary" id="btnHubSearch">Search HF</button>
          </div>
          <div class="model-list" id="hfResults">
            <div class="model-item">
              <div>
                <div class="model-name">Click "Search HF" to query Hugging Face GGUF catalog</div>
                <div class="model-meta">Includes graceful offline fallback to curated sovereign models.</div>
              </div>
            </div>
          </div>

          <div class="card-title" style="margin-top:6px;">
            <span>Local &amp; Loaded Sovereign Models</span>
            <button class="btn sm" id="btnRefreshTags">Refresh /api/tags</button>
          </div>
          <div class="model-list" id="localModels"></div>
        </div>

        <div class="card">
          <div class="card-title">
            <span>Ollama Modelfile Parser</span>
            <span class="badge">/api/studio/modelfile</span>
          </div>
          <label class="field">
            <span>Modelfile Source</span>
            <textarea id="modelfileInput" class="mono" rows="10">FROM qwen2.5-coder-7b-q4_k_m.gguf
PARAMETER temperature 0.4
PARAMETER num_ctx 8192
PARAMETER num_gpu 99
ADAPTER .b8g-cas/lora/error-repair.b8p
SYSTEM "You are a sovereign B8G TypeScript &amp; systems engineer."
TEMPLATE "{{ .System }}\n{{ .Prompt }}"</textarea>
          </label>
          <div style="display:flex;gap:8px;">
            <button class="btn primary" id="btnParseModelfile">Parse Modelfile</button>
            <button class="btn" id="btnApplyModelfile">Apply to Chat Tab</button>
          </div>
          <pre class="code-box" id="modelfileOutput" style="flex:1;min-height:150px;">Parsed OllamaModelfileSpec will appear here.</pre>
        </div>
      </div>
    </section>

    <!-- TAB 5: UNIFIED API SERVER -->
    <section class="tab-pane" id="pane-api">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>OpenAI-Compatible Endpoints (/v1/*)</span>
            <button class="btn sm ok" id="btnPingApi">Ping Live Server</button>
          </div>
          <table class="ep-table">
            <thead>
              <tr><th>Method</th><th>Endpoint</th><th>Purpose</th></tr>
            </thead>
            <tbody>
              <tr><td><span class="method-pill method-get">GET</span></td><td><code>/v1/models</code></td><td>List loaded GGUF &amp; embedding models</td></tr>
              <tr><td><span class="method-pill method-post">POST</span></td><td><code>/v1/chat/completions</code></td><td>Chat completions (SSE stream &amp; JSON)</td></tr>
              <tr><td><span class="method-pill method-post">POST</span></td><td><code>/v1/embeddings</code></td><td>Matryoshka + Jina-v3 task embeddings</td></tr>
            </tbody>
          </table>

          <div class="field-row">
            <span style="font-size:11px;color:var(--muted);font-weight:600;">OpenAI curl Snippet</span>
            <button class="btn sm" data-copy="snippetOpenAiCurl">Copy curl</button>
          </div>
          <pre class="code-box" id="snippetOpenAiCurl"></pre>

          <div class="field-row">
            <span style="font-size:11px;color:var(--muted);font-weight:600;">OpenAI fetch() Snippet</span>
            <button class="btn sm" data-copy="snippetOpenAiFetch">Copy fetch</button>
          </div>
          <pre class="code-box" id="snippetOpenAiFetch"></pre>
        </div>

        <div class="card">
          <div class="card-title">
            <span>Ollama-Compatible Endpoints (/api/*)</span>
            <span id="apiHealthBadge" class="badge">Unchecked</span>
          </div>
          <table class="ep-table">
            <thead>
              <tr><th>Method</th><th>Endpoint</th><th>Purpose</th></tr>
            </thead>
            <tbody>
              <tr><td><span class="method-pill method-get">GET</span></td><td><code>/api/tags</code></td><td>Ollama local model inventory</td></tr>
              <tr><td><span class="method-pill method-post">POST</span></td><td><code>/api/chat</code></td><td>Ollama chat protocol</td></tr>
              <tr><td><span class="method-pill method-post">POST</span></td><td><code>/api/embeddings</code></td><td>Ollama embedding vector endpoint</td></tr>
            </tbody>
          </table>

          <div class="field-row">
            <span style="font-size:11px;color:var(--muted);font-weight:600;">Ollama curl Snippet</span>
            <button class="btn sm" data-copy="snippetOllamaCurl">Copy curl</button>
          </div>
          <pre class="code-box" id="snippetOllamaCurl"></pre>

          <div class="field-row">
            <span style="font-size:11px;color:var(--muted);font-weight:600;">Ollama fetch() Snippet</span>
            <button class="btn sm" data-copy="snippetOllamaFetch">Copy fetch</button>
          </div>
          <pre class="code-box" id="snippetOllamaFetch"></pre>

          <pre class="code-box" id="apiPingOutput" style="min-height:64px;">Click "Ping Live Server" to verify /v1/models and /api/tags.</pre>
        </div>
      </div>
    </section>
  </main>
</div>

<script>
(function() {
  var $ = function(id) { return document.getElementById(id); };
  var origin = window.location.origin;

  /* ---------- Liveness 3px Progress Bar + Elapsed-ms Spinner Pill ---------- */
  var activeRequests = 0;
  var livenessStart = 0;
  var livenessTimer = null;

  function startLiveness(label) {
    activeRequests += 1;
    if (activeRequests === 1) {
      livenessStart = performance.now();
      $('livenessBar').classList.add('active');
      $('livenessPill').classList.add('active');
      if (livenessTimer) clearInterval(livenessTimer);
      livenessTimer = setInterval(function() {
        var ms = Math.round(performance.now() - livenessStart);
        $('livenessMs').textContent = ms + ' ms';
      }, 33);
    }
    $('livenessText').textContent = label || 'Working...';
  }

  function stopLiveness() {
    activeRequests = Math.max(0, activeRequests - 1);
    if (activeRequests === 0) {
      if (livenessTimer) {
        clearInterval(livenessTimer);
        livenessTimer = null;
      }
      $('livenessBar').classList.remove('active');
      $('livenessPill').classList.remove('active');
    }
  }

  async function withLiveness(label, fn) {
    startLiveness(label);
    try {
      return await fn();
    } finally {
      stopLiveness();
    }
  }

  /* ---------- Tab Switching ---------- */
  var tabButtons = document.querySelectorAll('.tab-btn');
  tabButtons.forEach(function(btn) {
    btn.addEventListener('click', function() {
      var target = btn.getAttribute('data-tab');
      tabButtons.forEach(function(b) { b.classList.remove('active'); });
      btn.classList.add('active');
      document.querySelectorAll('.tab-pane').forEach(function(p) { p.classList.remove('active'); });
      var pane = $('pane-' + target);
      if (pane) pane.classList.add('active');
      if (target === 'lora') drawLossCurve(lossHistory);
    });
  });

  function copyText(text, btn) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).catch(function() {});
    }
    if (btn) {
      var orig = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(function() { btn.textContent = orig; }, 1100);
    }
  }

  /* ---------- TAB 1: CHAT & LLAMA-SERVER CLI PREVIEW ---------- */
  var chatMessages = [];
  var chatAbort = null;

  function updateLlamaPreview() {
    var model = $('chatModel').value;
    var temp = parseFloat($('slTemp').value).toFixed(2);
    var ctx = $('slCtx').value;
    var ngl = $('slNgl').value;
    $('valTemp').textContent = temp;
    $('valCtx').textContent = ctx;
    $('valNgl').textContent = ngl;

    var isEmbed = model.toLowerCase().indexOf('embed') !== -1 || model.toLowerCase().indexOf('bert') !== -1;
    var parts = [
      'llama-server',
      '-m .b8g-cas/models/' + model + '.gguf',
      '--host 127.0.0.1',
      '--port 8080',
      '-c ' + ctx,
      '-ngl ' + ngl,
      '--flash-attn',
      '--temp ' + temp
    ];
    if (isEmbed) parts.push('--embedding');
    $('llamaCmdPreview').textContent = parts.join(' ');
  }

  ['chatModel', 'slTemp', 'slCtx', 'slNgl'].forEach(function(id) {
    $(id).addEventListener('input', updateLlamaPreview);
    $(id).addEventListener('change', updateLlamaPreview);
  });
  updateLlamaPreview();

  $('btnCopyLlama').addEventListener('click', function() {
    copyText($('llamaCmdPreview').textContent, $('btnCopyLlama'));
  });

  function appendChatBubble(role, text) {
    var wrap = document.createElement('div');
    wrap.className = 'msg ' + (role === 'user' ? 'user' : 'assistant');
    var roleEl = document.createElement('div');
    roleEl.className = 'msg-role';
    roleEl.textContent = role;
    var bodyEl = document.createElement('div');
    bodyEl.textContent = text;
    wrap.appendChild(roleEl);
    wrap.appendChild(bodyEl);
    var box = $('chatTranscript');
    box.appendChild(wrap);
    box.scrollTop = box.scrollHeight;
    return bodyEl;
  }

  async function sendChat() {
    var prompt = $('chatPrompt').value.trim();
    if (!prompt) return;
    $('chatPrompt').value = '';
    appendChatBubble('user', prompt);
    chatMessages.push({ role: 'user', content: prompt });

    var assistantEl = appendChatBubble('assistant', '');
    var fullReply = '';
    chatAbort = new AbortController();
    $('btnSendChat').disabled = true;
    $('btnStopChat').disabled = false;

    try {
      await withLiveness('Streaming SSE chat...', async function() {
        var payload = {
          model: $('chatModel').value,
          stream: true,
          temperature: parseFloat($('slTemp').value),
          max_tokens: 512,
          messages: [{ role: 'system', content: $('chatSystem').value }].concat(chatMessages)
        };
        var res = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: chatAbort.signal
        });
        if (!res.ok) {
          throw new Error('HTTP ' + res.status + ' ' + (await res.text()));
        }
        var ct = (res.headers.get('content-type') || '').toLowerCase();
        if (ct.indexOf('text/event-stream') !== -1 && res.body) {
          var reader = res.body.getReader();
          var decoder = new TextDecoder();
          var buf = '';
          while (true) {
            var chunk = await reader.read();
            if (chunk.done) break;
            buf += decoder.decode(chunk.value, { stream: true });
            var lines = buf.split('\\n');
            buf = lines.pop() || '';
            for (var i = 0; i < lines.length; i++) {
              var line = lines[i].trim();
              if (!line.startsWith('data:')) continue;
              var data = line.slice(5).trim();
              if (!data || data === '[DONE]') continue;
              try {
                var parsed = JSON.parse(data);
                var delta = parsed.choices && parsed.choices[0] && parsed.choices[0].delta && parsed.choices[0].delta.content;
                var msg = parsed.choices && parsed.choices[0] && parsed.choices[0].message && parsed.choices[0].message.content;
                var tok = delta || msg || '';
                if (tok) {
                  fullReply += tok;
                  assistantEl.textContent = fullReply;
                  $('chatTranscript').scrollTop = $('chatTranscript').scrollHeight;
                }
              } catch (_e) {}
            }
          }
        } else {
          var json = await res.json();
          fullReply = (json.choices && json.choices[0] && json.choices[0].message && json.choices[0].message.content) ||
            json.response || JSON.stringify(json, null, 2);
          assistantEl.textContent = fullReply;
        }
      });
      if (fullReply) chatMessages.push({ role: 'assistant', content: fullReply });
    } catch (err) {
      if (err && err.name === 'AbortError') {
        assistantEl.textContent = (fullReply || '') + ' [stopped]';
      } else {
        assistantEl.textContent = 'Error: ' + (err && err.message ? err.message : String(err));
      }
    } finally {
      chatAbort = null;
      $('btnSendChat').disabled = false;
      $('btnStopChat').disabled = true;
    }
  }

  $('btnSendChat').addEventListener('click', sendChat);
  $('chatPrompt').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      sendChat();
    }
  });
  $('btnStopChat').addEventListener('click', function() {
    if (chatAbort) chatAbort.abort();
  });
  $('btnClearChat').addEventListener('click', function() {
    chatMessages = [];
    $('chatTranscript').innerHTML = '';
    appendChatBubble('system', 'Transcript cleared.');
  });

  /* ---------- TAB 2: EMBEDDINGS & MATRYOSHKA LAB ---------- */
  var selectedDim = 64;
  document.querySelectorAll('#dimGroup .seg-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      document.querySelectorAll('#dimGroup .seg-btn').forEach(function(b) { b.classList.remove('active'); });
      btn.classList.add('active');
      selectedDim = parseInt(btn.getAttribute('data-dim') || '64', 10);
      $('embedDimBadge').textContent = selectedDim + ' dims';
    });
  });

  function l2Norm(vec) {
    var s = 0;
    for (var i = 0; i < vec.length; i++) s += vec[i] * vec[i];
    return Math.sqrt(s);
  }

  function cosineSim(a, b) {
    var n = Math.min(a.length, b.length);
    var dot = 0, na = 0, nb = 0;
    for (var i = 0; i < n; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    var denom = Math.sqrt(na) * Math.sqrt(nb);
    return denom > 0 ? dot / denom : 0;
  }

  function renderSparkline(containerId, vec) {
    var el = $(containerId);
    el.innerHTML = '';
    var maxAbs = 0.001;
    for (var i = 0; i < vec.length; i++) {
      var a = Math.abs(vec[i]);
      if (a > maxAbs) maxAbs = a;
    }
    for (var j = 0; j < vec.length; j++) {
      var bar = document.createElement('div');
      bar.className = 'vec-bar';
      var h = Math.max(8, Math.round((Math.abs(vec[j]) / maxAbs) * 100));
      bar.style.height = h + '%';
      bar.style.background = vec[j] >= 0 ? '#38bdf8' : '#a78bfa';
      bar.title = '[' + j + '] ' + vec[j].toFixed(4);
      el.appendChild(bar);
    }
  }

  $('btnComputeEmbed').addEventListener('click', async function() {
    var textA = $('embedTextA').value.trim();
    var textB = $('embedTextB').value.trim();
    var adapter = $('embedAdapter').value;
    if (!textA || !textB) return;

    try {
      await withLiveness('Computing ' + selectedDim + 'd embeddings...', async function() {
        var res = await fetch('/v1/embeddings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: 'jina-embeddings-v3-gguf',
            input: [textA, textB],
            task: adapter,
            task_id: adapter,
            dimensions: selectedDim
          })
        });
        if (!res.ok) throw new Error('HTTP ' + res.status + ' ' + (await res.text()));
        var json = await res.json();
        var data = json.data || [];
        var vecA = (data[0] && data[0].embedding) || [];
        var vecB = (data[1] && data[1].embedding) || [];
        var sim = typeof json.cosineSimilarity === 'number' ? json.cosineSimilarity : cosineSim(vecA, vecB);
        var pct = Math.max(0, Math.min(100, Math.round(((sim + 1) / 2) * 100)));

        $('simScore').textContent = sim.toFixed(4);
        $('normA').textContent = l2Norm(vecA).toFixed(4);
        $('normB').textContent = l2Norm(vecB).toFixed(4);
        $('simFill').style.width = pct + '%';
        $('simPctLabel').textContent = pct + '% (' + adapter + ')';
        $('embedStatus').textContent = 'OK - ' + vecA.length + 'd Matryoshka vectors';
        renderSparkline('vecBarsA', vecA);
        renderSparkline('vecBarsB', vecB);

        var fmt = function(v) {
          return '[' + v.slice(0, 12).map(function(x) { return x.toFixed(4); }).join(', ') + (v.length > 12 ? ', ...' : '') + ']';
        };
        $('vecPreview').textContent =
          'Adapter: ' + adapter + ' | Dim: ' + vecA.length + '\\n' +
          'Vec A: ' + fmt(vecA) + '\\n' +
          'Vec B: ' + fmt(vecB);
      });
    } catch (err) {
      $('embedStatus').textContent = 'Error: ' + (err && err.message ? err.message : String(err));
    }
  });

  /* ---------- TAB 3: LORA & UNSLOTH FINE-TUNER + CANVAS 2D LOSS CURVE ---------- */
  var loraStrategy = 'direct-task-finetune';
  var loraQuant = 'nf4';
  var lossHistory = [];
  var lastAdapterId = null;

  document.querySelectorAll('#loraStrategyGroup .seg-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      document.querySelectorAll('#loraStrategyGroup .seg-btn').forEach(function(b) { b.classList.remove('active'); });
      btn.classList.add('active');
      loraStrategy = btn.getAttribute('data-strategy') || 'direct-task-finetune';
    });
  });

  function updateVramEstimate() {
    var paramsB = 7;
    var fp16Base = Math.round(paramsB * 2000);
    var fp16Opt = Math.round(fp16Base * 0.28);
    var fp16Total = fp16Base + fp16Opt;
    var mult = loraQuant === 'nf4' ? 0.27 : (loraQuant === 'q8' ? 0.52 : 1.0);
    var baseMb = Math.round(fp16Base * mult);
    var loraOptMb = loraQuant === 'nf4' ? 420 : (loraQuant === 'q8' ? 680 : fp16Opt);
    var totalMb = baseMb + loraOptMb;
    var savePct = Math.max(0, Math.round((1 - totalMb / fp16Total) * 100));

    $('vramBase').textContent = baseMb + ' MB';
    $('vramLora').textContent = loraOptMb + ' MB';
    $('vramTotal').textContent = totalMb + ' MB';
    $('vramSave').textContent = savePct > 0 ? ('-' + savePct + '%') : '0% (base)';
  }

  document.querySelectorAll('#loraQuantGroup .seg-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      document.querySelectorAll('#loraQuantGroup .seg-btn').forEach(function(b) { b.classList.remove('active'); });
      btn.classList.add('active');
      loraQuant = btn.getAttribute('data-quant') || 'nf4';
      updateVramEstimate();
    });
  });
  updateVramEstimate();

  function drawLossCurve(points) {
    var canvas = $('lossCanvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    if (!ctx) return;

    var w = canvas.width;
    var h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    var padL = 50, padR = 20, padT = 22, padB = 30;
    var plotW = w - padL - padR;
    var plotH = h - padT - padB;

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
    ctx.lineWidth = 1;
    for (var g = 0; g <= 4; g++) {
      var gy = padT + (plotH * g) / 4;
      ctx.beginPath();
      ctx.moveTo(padL, gy);
      ctx.lineTo(w - padR, gy);
      ctx.stroke();
    }

    if (!points || points.length === 0) {
      ctx.fillStyle = '#94a3b8';
      ctx.font = '12px ui-monospace, Consolas, monospace';
      ctx.fillText('No training steps yet. Click "Train LoRA" to plot live contrastive loss.', padL + 12, padT + plotH / 2);
      return;
    }

    var minL = points[0];
    var maxL = points[0];
    for (var i = 1; i < points.length; i++) {
      if (points[i] < minL) minL = points[i];
      if (points[i] > maxL) maxL = points[i];
    }
    var span = Math.max(0.0001, maxL - minL);
    var yMin = Math.max(0, minL - span * 0.12);
    var yMax = maxL + span * 0.12;

    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px ui-monospace, Consolas, monospace';
    ctx.fillText(yMax.toFixed(3), 6, padT + 4);
    ctx.fillText(((yMax + yMin) / 2).toFixed(3), 6, padT + plotH / 2 + 4);
    ctx.fillText(yMin.toFixed(3), 6, padT + plotH + 4);
    ctx.fillText('Step 1', padL, h - 10);
    ctx.fillText('Step ' + points.length, w - padR - 48, h - 10);

    var coords = points.map(function(val, idx) {
      var x = padL + (points.length === 1 ? plotW / 2 : (idx / (points.length - 1)) * plotW);
      var y = padT + ((yMax - val) / Math.max(0.0001, yMax - yMin)) * plotH;
      return { x: x, y: y };
    });

    var grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
    grad.addColorStop(0, 'rgba(56, 189, 248, 0.32)');
    grad.addColorStop(1, 'rgba(167, 139, 250, 0.02)');
    ctx.beginPath();
    ctx.moveTo(coords[0].x, padT + plotH);
    for (var k = 0; k < coords.length; k++) ctx.lineTo(coords[k].x, coords[k].y);
    ctx.lineTo(coords[coords.length - 1].x, padT + plotH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    var lineGrad = ctx.createLinearGradient(padL, 0, w - padR, 0);
    lineGrad.addColorStop(0, '#a78bfa');
    lineGrad.addColorStop(1, '#34d399');
    ctx.strokeStyle = lineGrad;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (var m = 0; m < coords.length; m++) {
      if (m === 0) ctx.moveTo(coords[m].x, coords[m].y);
      else ctx.lineTo(coords[m].x, coords[m].y);
    }
    ctx.stroke();

    var last = coords[coords.length - 1];
    ctx.fillStyle = '#34d399';
    ctx.beginPath();
    ctx.arc(last.x, last.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  drawLossCurve([]);

  function recordTrainEvent(ev) {
    if (typeof ev.loss === 'number') {
      lossHistory.push(ev.loss);
      drawLossCurve(lossHistory);
      $('kpiLoss').textContent = ev.loss.toFixed(4);
      var delta = lossHistory.length > 1 ? (ev.loss - lossHistory[0]) : 0;
      $('kpiDelta').textContent = (delta <= 0 ? '' : '+') + delta.toFixed(4);
    }
    if (typeof ev.posSim === 'number') $('kpiPosSim').textContent = ev.posSim.toFixed(4);
    if (typeof ev.negSim === 'number') $('kpiNegSim').textContent = ev.negSim.toFixed(4);
    if (typeof ev.step === 'number') {
      $('loraStepBadge').textContent = 'Step ' + ev.step + (ev.totalSteps ? '/' + ev.totalSteps : '');
    }
  }

  $('btnTrainLora').addEventListener('click', async function() {
    var pairs;
    try {
      pairs = JSON.parse($('loraPairs').value);
      if (!Array.isArray(pairs) || pairs.length === 0) throw new Error('Provide at least 1 triplet pair');
    } catch (err) {
      $('loraLog').textContent = 'Invalid JSON training pairs: ' + err.message;
      return;
    }

    lossHistory = [];
    drawLossCurve([]);
    $('loraLog').textContent = 'Starting ' + $('loraRecipe').value + ' (' + loraStrategy + ', ' + loraQuant.toUpperCase() + ')...\\n';
    $('loraStepBadge').textContent = 'Training...';

    try {
      await withLiveness('Training LoRA adapter...', async function() {
        var res = await fetch('/api/studio/lora/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            taskId: $('loraTask').value,
            strategy: loraStrategy,
            recipe: $('loraRecipe').value,
            quant: loraQuant,
            steps: 24,
            lr: 0.08,
            pairs: pairs
          })
        });
        if (!res.ok) throw new Error('HTTP ' + res.status + ' ' + (await res.text()));
        var ct = (res.headers.get('content-type') || '').toLowerCase();
        if ((ct.indexOf('event-stream') !== -1 || ct.indexOf('ndjson') !== -1) && res.body) {
          var reader = res.body.getReader();
          var decoder = new TextDecoder();
          var buf = '';
          while (true) {
            var chunk = await reader.read();
            if (chunk.done) break;
            buf += decoder.decode(chunk.value, { stream: true });
            var lines = buf.split('\\n');
            buf = lines.pop() || '';
            for (var i = 0; i < lines.length; i++) {
              var raw = lines[i].trim();
              if (!raw) continue;
              if (raw.startsWith('data:')) raw = raw.slice(5).trim();
              if (!raw || raw === '[DONE]') continue;
              try {
                var ev = JSON.parse(raw);
                recordTrainEvent(ev);
                if (ev.adapterId) lastAdapterId = ev.adapterId;
              } catch (_e) {}
            }
          }
        } else {
          var json = await res.json();
          if (json.adapterId) lastAdapterId = json.adapterId;
          if (json.vram && typeof json.vram.totalMb === 'number') {
            $('vramBase').textContent = json.vram.baseMb + ' MB';
            $('vramLora').textContent = (json.vram.loraMb + json.vram.optimizerMb) + ' MB';
            $('vramTotal').textContent = json.vram.totalMb + ' MB';
            $('vramSave').textContent = '-' + Math.round(json.vram.savingsPctVsFp16) + '%';
          }
          var steps = json.steps || json.events || [];
          if (Array.isArray(steps) && steps.length > 0) {
            steps.forEach(recordTrainEvent);
          } else if (Array.isArray(json.lossHistory)) {
            json.lossHistory.forEach(function(l, idx) {
              recordTrainEvent({ step: idx + 1, totalSteps: json.lossHistory.length, loss: l });
            });
          }
          $('loraLog').textContent += JSON.stringify(json, null, 2);
        }
        $('loraStepBadge').textContent = 'Trained (' + lossHistory.length + ' steps)';
      });
    } catch (err) {
      $('loraStepBadge').textContent = 'Error';
      $('loraLog').textContent += 'Training failed: ' + (err && err.message ? err.message : String(err));
    }
  });

  $('btnMergeLora').addEventListener('click', async function() {
    try {
      await withLiveness('Merging W\\\' = W + B*A to CAS...', async function() {
        var res = await fetch('/api/studio/lora/merge', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            taskId: $('loraTask').value,
            strategy: loraStrategy,
            adapterId: lastAdapterId
          })
        });
        if (!res.ok) throw new Error('HTTP ' + res.status + ' ' + (await res.text()));
        var json = await res.json();
        var digest = json.sha256 || json.digest || 'merged';
        $('casDigestBadge').textContent = 'CAS ' + String(digest).slice(0, 12);
        $('casDigestBadge').className = 'badge ok';
        $('loraLog').textContent = 'Merged W\\\' = W + (alpha/rank) * B * A into CAS:\\n' + JSON.stringify(json, null, 2);
      });
    } catch (err) {
      $('loraLog').textContent = 'Merge failed: ' + (err && err.message ? err.message : String(err));
    }
  });

  /* ---------- TAB 4: MODEL HUB (HF + OLLAMA + LOCAL GGUF) ---------- */
  function renderModelRow(container, m, badgeText) {
    var row = document.createElement('div');
    row.className = 'model-item';
    var id = m.id || m.name || m.model || 'unknown-model';
    var quant = m.quant || m.quantization || (m.details && m.details.quantization_level) || 'GGUF';
    var metaParts = [quant];
    if (m.kind) metaParts.push(m.kind);
    if (typeof m.downloads === 'number') metaParts.push(m.downloads.toLocaleString() + ' dl');
    if (typeof m.size === 'number') metaParts.push((m.size / (1024 * 1024)).toFixed(1) + ' MB');

    var left = document.createElement('div');
    var nameEl = document.createElement('div');
    nameEl.className = 'model-name';
    nameEl.textContent = id;
    var metaEl = document.createElement('div');
    metaEl.className = 'model-meta';
    metaEl.textContent = metaParts.join(' · ');
    left.appendChild(nameEl);
    left.appendChild(metaEl);

    var right = document.createElement('div');
    right.style.display = 'flex';
    right.style.gap = '6px';
    right.style.alignItems = 'center';
    if (badgeText) {
      var b = document.createElement('span');
      b.className = 'badge sky';
      b.textContent = badgeText;
      right.appendChild(b);
    }
    var useBtn = document.createElement('button');
    useBtn.className = 'btn sm';
    useBtn.textContent = 'Use in Chat';
    useBtn.addEventListener('click', function() {
      var sel = $('chatModel');
      var exists = false;
      for (var i = 0; i < sel.options.length; i++) {
        if (sel.options[i].value === id) exists = true;
      }
      if (!exists) {
        var opt = document.createElement('option');
        opt.value = id;
        opt.textContent = id;
        sel.appendChild(opt);
      }
      sel.value = id;
      updateLlamaPreview();
      document.querySelector('.tab-btn[data-tab="chat"]').click();
    });
    right.appendChild(useBtn);

    row.appendChild(left);
    row.appendChild(right);
    container.appendChild(row);
  }

  async function searchHub() {
    var q = $('hubQuery').value.trim();
    try {
      await withLiveness('Searching HF GGUF Hub...', async function() {
        var res = await fetch('/api/studio/hub/search?q=' + encodeURIComponent(q) + '&limit=12');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        var json = await res.json();
        var items = Array.isArray(json) ? json : (json.models || json.results || []);
        var box = $('hfResults');
        box.innerHTML = '';
        if (items.length === 0) {
          box.innerHTML = '<div class="model-item"><div class="model-meta">No matching GGUF models found.</div></div>';
          return;
        }
        items.forEach(function(m) {
          renderModelRow(box, m, m.source || 'HF');
        });
      });
    } catch (err) {
      $('hfResults').innerHTML = '<div class="model-item"><div class="model-meta">Search error: ' + String(err.message || err) + '</div></div>';
    }
  }

  async function loadLocalTags() {
    try {
      await withLiveness('Loading /api/tags...', async function() {
        var res = await fetch('/api/tags');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        var json = await res.json();
        var models = json.models || [];
        var box = $('localModels');
        box.innerHTML = '';
        var sel = $('chatModel');
        models.forEach(function(m) {
          renderModelRow(box, m, 'LOCAL');
          var id = m.name || m.model || m.id;
          if (id) {
            var found = false;
            for (var i = 0; i < sel.options.length; i++) {
              if (sel.options[i].value === id) found = true;
            }
            if (!found) {
              var opt = document.createElement('option');
              opt.value = id;
              opt.textContent = id;
              sel.appendChild(opt);
            }
          }
        });
        if (models.length === 0) {
          box.innerHTML = '<div class="model-item"><div class="model-meta">No local models reported by /api/tags.</div></div>';
        }
      });
    } catch (err) {
      $('localModels').innerHTML = '<div class="model-item"><div class="model-meta">Unable to fetch /api/tags: ' + String(err.message || err) + '</div></div>';
    }
  }

  $('btnHubSearch').addEventListener('click', searchHub);
  $('hubQuery').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') searchHub();
  });
  $('btnRefreshTags').addEventListener('click', loadLocalTags);

  var lastParsedModelfile = null;
  $('btnParseModelfile').addEventListener('click', async function() {
    var text = $('modelfileInput').value;
    try {
      await withLiveness('Parsing Ollama Modelfile...', async function() {
        var res = await fetch('/api/studio/modelfile', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text, modelfile: text })
        });
        if (!res.ok) throw new Error('HTTP ' + res.status + ' ' + (await res.text()));
        var json = await res.json();
        lastParsedModelfile = json;
        $('modelfileOutput').textContent = JSON.stringify(json, null, 2);
      });
    } catch (err) {
      $('modelfileOutput').textContent = 'Parse error: ' + (err && err.message ? err.message : String(err));
    }
  });

  $('btnApplyModelfile').addEventListener('click', function() {
    if (!lastParsedModelfile) return;
    var spec = lastParsedModelfile.spec || lastParsedModelfile;
    if (spec.from) {
      var clean = String(spec.from).replace(/\\.gguf$/i, '');
      var sel = $('chatModel');
      var exists = false;
      for (var i = 0; i < sel.options.length; i++) {
        if (sel.options[i].value === clean) exists = true;
      }
      if (!exists) {
        var opt = document.createElement('option');
        opt.value = clean;
        opt.textContent = clean;
        sel.appendChild(opt);
      }
      sel.value = clean;
    }
    if (spec.system) $('chatSystem').value = spec.system;
    if (spec.parameters) {
      if (spec.parameters.temperature !== undefined) $('slTemp').value = String(spec.parameters.temperature);
      if (spec.parameters.num_ctx !== undefined) $('slCtx').value = String(spec.parameters.num_ctx);
      if (spec.parameters.num_gpu !== undefined) $('slNgl').value = String(spec.parameters.num_gpu);
    }
    updateLlamaPreview();
    document.querySelector('.tab-btn[data-tab="chat"]').click();
  });

  /* ---------- TAB 5: UNIFIED API SERVER SNIPPETS ---------- */
  $('snippetOpenAiCurl').textContent =
    'curl ' + origin + '/v1/chat/completions \\\\\\n' +
    '  -H "Content-Type: application/json" \\\\\\n' +
    '  -d \\\'{"model":"qwen2.5-coder-7b-q4_k_m","stream":true,"messages":[{"role":"user","content":"Hello"}]}\\\'';

  $('snippetOpenAiFetch').textContent =
    'const res = await fetch("' + origin + '/v1/embeddings", {\\n' +
    '  method: "POST",\\n' +
    '  headers: { "Content-Type": "application/json" },\\n' +
    '  body: JSON.stringify({ model: "jina-embeddings-v3-gguf", input: ["hello"], task: "text-matching", dimensions: 64 })\\n' +
    '});\\n' +
    'console.log(await res.json());';

  $('snippetOllamaCurl').textContent =
    'curl ' + origin + '/api/chat \\\\\\n' +
    '  -H "Content-Type: application/json" \\\\\\n' +
    '  -d \\\'{"model":"llama-3.2-3b-instruct-q4_k_m","messages":[{"role":"user","content":"Ping"}]}\\\'';

  $('snippetOllamaFetch').textContent =
    'const tags = await fetch("' + origin + '/api/tags").then(r => r.json());\\n' +
    'const emb = await fetch("' + origin + '/api/embeddings", {\\n' +
    '  method: "POST",\\n' +
    '  headers: { "Content-Type": "application/json" },\\n' +
    '  body: JSON.stringify({ model: "lumiBERT-local", prompt: "sovereign cas" })\\n' +
    '}).then(r => r.json());\\n' +
    'console.log({ tags, emb });';

  document.querySelectorAll('[data-copy]').forEach(function(btn) {
    btn.addEventListener('click', function() {
      var targetId = btn.getAttribute('data-copy');
      var el = $(targetId);
      if (el) copyText(el.textContent, btn);
    });
  });

  $('btnPingApi').addEventListener('click', async function() {
    try {
      await withLiveness('Pinging /v1/models & /api/tags...', async function() {
        var r1 = await fetch('/v1/models').then(function(r) { return r.json(); });
        var r2 = await fetch('/api/tags').then(function(r) { return r.json(); });
        $('apiHealthBadge').textContent = 'ONLINE';
        $('apiHealthBadge').className = 'badge ok';
        $('apiPingOutput').textContent = JSON.stringify({
          openaiModelsCount: (r1.data || []).length,
          ollamaTagsCount: (r2.models || []).length,
          sampleOpenAi: (r1.data && r1.data[0]) || null
        }, null, 2);
      });
    } catch (err) {
      $('apiHealthBadge').textContent = 'OFFLINE';
      $('apiHealthBadge').className = 'badge warn';
      $('apiPingOutput').textContent = 'Ping error: ' + (err && err.message ? err.message : String(err));
    }
  });

  /* Initial silent catalog sync */
  loadLocalTags();
})();
</script>
</body>
</html>`;
}
