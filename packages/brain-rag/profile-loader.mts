/**
 * B8G-Brain 1:1 Profile Loader & Session Scaffolder.
 *
 * Reads `.agents/rules/*.md` and `.agents/skills/<name>/SKILL.md` (YAML frontmatter),
 * parses `transcript.jsonl` trajectories, and scaffolds 1:1 compatible
 * `brain/<session-id>/` session directories.
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, appendFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';

export interface LoadedRule {
  name: string;
  path: string;
  content: string;
}

export interface LoadedSkill {
  name: string;
  description: string;
  path: string;
  body: string;
}

export interface TranscriptStep {
  step_index: number;
  source: 'USER_EXPLICIT' | 'MODEL' | 'SYSTEM';
  type: string;
  status: string;
  created_at: string;
  content?: string;
  thinking?: string;
  tool_calls?: unknown[];
  media?: Array<{ mime_type: string; uri: string }>;
  truncated_fields?: string[];
}

function parseFrontmatter(raw: string): { meta: Record<string, string>; body: string } {
  const m = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
  if (!m) return { meta: {}, body: raw };
  const meta: Record<string, string> = {};
  for (const line of m[1].split(/\r?\n/)) {
    const kv = line.match(/^([a-zA-Z0-9_-]+):\s*(.+)$/);
    if (kv) meta[kv[1].trim()] = kv[2].trim().replace(/^["']|["']$/g, '');
  }
  return { meta, body: m[2].trim() };
}

export function loadWorkspaceProfile(workspaceRoot: string): {
  rules: LoadedRule[];
  skills: LoadedSkill[];
} {
  const agentsDir = join(resolve(workspaceRoot), '.agents');
  const rules: LoadedRule[] = [];
  const skills: LoadedSkill[] = [];

  const rulesDir = join(agentsDir, 'rules');
  if (existsSync(rulesDir)) {
    for (const f of readdirSync(rulesDir).sort()) {
      if (!f.endsWith('.md')) continue;
      const p = join(rulesDir, f);
      rules.push({ name: basename(f, '.md'), path: p, content: readFileSync(p, 'utf8') });
    }
  }

  const skillsDir = join(agentsDir, 'skills');
  if (existsSync(skillsDir)) {
    for (const dir of readdirSync(skillsDir, { withFileTypes: true })) {
      if (!dir.isDirectory()) continue;
      const skillFile = join(skillsDir, dir.name, 'SKILL.md');
      if (!existsSync(skillFile)) continue;
      const { meta, body } = parseFrontmatter(readFileSync(skillFile, 'utf8'));
      skills.push({
        name: meta.name || dir.name,
        description: meta.description || '',
        path: skillFile,
        body,
      });
    }
  }

  return { rules, skills };
}

/** Scaffolds a 1:1 compatible `brain/<sessionId>/` directory layout. */
export function scaffoldBrainSessionDir(brainRoot: string, sessionId: string): string {
  const root = join(resolve(brainRoot), sessionId);
  const dirs = [
    root,
    join(root, 'scratch'),
    join(root, '.user_uploaded'),
    join(root, '.system_generated', 'logs'),
    join(root, '.system_generated', 'steps'),
    join(root, '.system_generated', 'tasks'),
    join(root, '.system_generated', 'messages', 'undelivered'),
    join(root, '.system_generated', 'subagents'),
    join(root, '.system_generated', 'swarm'),
  ];
  for (const d of dirs) mkdirSync(d, { recursive: true });
  return root;
}

export function appendTranscriptStep(sessionDir: string, step: TranscriptStep): void {
  const logFile = join(sessionDir, '.system_generated', 'logs', 'transcript.jsonl');
  mkdirSync(join(sessionDir, '.system_generated', 'logs'), { recursive: true });
  appendFileSync(logFile, JSON.stringify(step) + '\n', 'utf8');
}

export function readTranscriptSteps(sessionDir: string): TranscriptStep[] {
  const logFile = join(sessionDir, '.system_generated', 'logs', 'transcript.jsonl');
  if (!existsSync(logFile)) return [];
  return readFileSync(logFile, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line) as TranscriptStep);
}
