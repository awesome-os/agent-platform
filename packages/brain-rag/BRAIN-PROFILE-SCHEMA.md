---
title: "Universal Brain & Profile Directory Schema (1:1 Alignment Spec)"
slug: "b8g-brain-profile-schema"
updated: "2026-11-20"
motto: "Nunquam ante, numquam iterum"
---

# Universal Brain & Profile Directory Schema

This document specifies the exact directory layout, file formats, and semantic content types used to align **`B8G-Brain`** (`intersections/b8g-brain/`) 1:1 with modern `.agents/` + `brain/<session-id>/` user profile directories.

---

## 1. Two-Layer Directory Topology

### Layer A — Workspace Customization Layer (`<workspace>/.agents/`)

| Path | Format | Semantic Role | Loading Policy |
| :--- | :--- | :--- | :--- |
| `.agents/rules/*.md` | Plain Markdown | Hard behavioral & architectural invariants (e.g., `bin-shim-virtualization.md`, `environment-shell.md`) | Always loaded into system prompt |
| `.agents/skills/<name>/SKILL.md` | YAML frontmatter (`name`, `description`) + Markdown body | On-demand domain capability instructions | Frontmatter indexed at startup; body loaded on match |
| `.agents/skills/<name>/references/` | Markdown / JSON / specs | Extended reference docs for a skill | Read on demand |
| `.agents/skills/<name>/scripts/` | `.mts` / `.ps1` / `.sh` | Executable helper scripts | Invoked on demand |

### Layer B — Per-Session Isolated Brain (`<profile>/brain/<session-id>/`)

| Path | Format | Semantic Role |
| :--- | :--- | :--- |
| `*.md` | GitHub-Flavored Markdown (KaTeX + Mermaid + alerts) | User-facing persistent artifacts (plans, reports, walkthroughs) |
| `scratch/` | `.mts`, `.ps1`, `.sh`, `.json`, temp data | Persistent per-session scratch scripts & intermediate outputs |
| `.user_uploaded/` | `media_<ts>_<hash>.{png,jpg,webp,pdf}` | Immutable user-attached media referenced by URI from transcript |
| `.system_generated/logs/transcript.jsonl` | Compact JSON Lines (`.jsonl`) | Event-sourced trajectory (large fields truncated with `truncated_fields`) |
| `.system_generated/logs/transcript_full.jsonl` | Full lossless JSON Lines (`.jsonl`) | Exact untruncated step log for replay & audit |
| `.system_generated/steps/<N>/output.txt` | Plain text / UTF-8 | Overflow spill storage for large tool outputs |
| `.system_generated/tasks/task-<N>.log` | Streaming UTF-8 log | Background task stdout/stderr log |
| `.system_generated/messages/<uuid>.json` | JSON (+ `undelivered/` queue) | Async inter-agent & background task delivery queue |
| `.system_generated/subagents/` | JSON state | Direct child subagent conversation registry |
| `.system_generated/swarm/` | JSONL / JSON | Multi-agent shared message board & roster state |

---

## 2. Exact Semantic Schemas

### 2.1 `SKILL.md` Frontmatter Grammar
```yaml
---
name: my-skill-name
description: Concise trigger description telling the brain when to load this skill.
---
# Skill Instructions (Markdown)
```

### 2.2 Trajectory Step Record (`transcript.jsonl` — 1 JSON object per line)
```json
{
  "step_index": 42,
  "source": "USER_EXPLICIT | MODEL | SYSTEM",
  "type": "USER_INPUT | PLANNER_RESPONSE | TOOL_RESULT",
  "status": "DONE | ERROR | RUNNING",
  "created_at": "2026-11-20T14:30:00Z",
  "content": "Text payload",
  "thinking": "Optional reasoning trace",
  "tool_calls": [{ "name": "run_command", "args": {} }],
  "media": [{ "mime_type": "image/png", "uri": "file:///..." }],
  "truncated_fields": ["content"]
}
```

### 2.3 Algebraic Mapping to `B8G-Brain`
- **Tenant (`tenants/<tenantId>/`)** = Workspace + `.agents/{rules,skills}` + Incremental SHA-256 CAS Code Index + Model-Independent Episodic Memory.
- **Session (`brain/<sessionId>/`)** = Isolated event-sourced `transcript.jsonl` + `scratch/` + `.user_uploaded/` + `tasks/` + `messages/` — **100% portable across any LLM (`OpenRouter`, `Ollama`, `Gemini`, `Claude`, `DeepSeek`)**.
