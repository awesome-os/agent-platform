/**
 * @file dashboards/build-dashboards.mts
 * @description Materializes static standalone HTML dashboards for AwesomeOS Agent Platform:
 *   - dashboards/llm-studio.html     (from ../packages/llm-studio/ui-html.mts)
 *   - dashboards/agent-host.html     (from ../packages/agent-host/ui-html.mts)
 *   - dashboards/sectium-shell.html  (from ../packages/agent-host/sectium-shell-html.mts)
 *   - dashboards/oversight.html      (from ../holo-v11/dashboard/oversight.html when mirrored)
 *   - dashboards/host-manager.html   (from ../host-manager/console/index.html when mirrored)
 *   - dashboards/index.html          (verified Unified Control Center shell)
 *
 * Strictly `erasableSyntaxOnly: true` & zero external dependencies.
 */

import fs from 'node:fs';
import path from 'node:path';
import { renderLlmStudioHtml } from '../packages/llm-studio/ui-html.mts';
import { renderAgentHostHtml } from '../packages/agent-host/ui-html.mts';
import { renderSectiumShellHtml } from '../packages/agent-host/sectium-shell-html.mts';

const DASHBOARDS_DIR = import.meta.dirname;
const PLATFORM_ROOT = path.resolve(DASHBOARDS_DIR, '..');
const MIRROR_ROOT = path.resolve(PLATFORM_ROOT, '..');

interface BuildArtifact {
  readonly name: string;
  readonly targetPath: string;
  readonly bytes: number;
  readonly source: string;
}

function writeDashboard(name: string, html: string, source: string): BuildArtifact {
  const targetPath = path.join(DASHBOARDS_DIR, name);
  fs.writeFileSync(targetPath, html, 'utf8');
  const bytes = Buffer.byteLength(html, 'utf8');
  return { name, targetPath, bytes, source };
}

function copyOrVerifyDashboard(
  name: string,
  candidateSources: readonly string[],
): BuildArtifact {
  const targetPath = path.join(DASHBOARDS_DIR, name);
  for (const src of candidateSources) {
    if (fs.existsSync(src)) {
      const html = fs.readFileSync(src, 'utf8');
      fs.writeFileSync(targetPath, html, 'utf8');
      return {
        name,
        targetPath,
        bytes: Buffer.byteLength(html, 'utf8'),
        source: `copied:${src}`,
      };
    }
  }
  if (fs.existsSync(targetPath)) {
    const stat = fs.statSync(targetPath);
    return {
      name,
      targetPath,
      bytes: stat.size,
      source: 'existing-standalone',
    };
  }
  throw new Error(
    `Missing dashboard "${name}" and none of candidate sources exist: ${candidateSources.join(', ')}`,
  );
}

function main(): void {
  fs.mkdirSync(DASHBOARDS_DIR, { recursive: true });

  const artifacts: BuildArtifact[] = [];

  artifacts.push(
    writeDashboard(
      'llm-studio.html',
      renderLlmStudioHtml(),
      'renderLlmStudioHtml() @ packages/llm-studio/ui-html.mts',
    ),
  );

  artifacts.push(
    writeDashboard(
      'agent-host.html',
      renderAgentHostHtml({ port: 7498, defaultProjectRoot: '/workspace' }),
      'renderAgentHostHtml() @ packages/agent-host/ui-html.mts',
    ),
  );

  artifacts.push(
    writeDashboard(
      'sectium-shell.html',
      renderSectiumShellHtml({ port: 7498, defaultProjectRoot: '/workspace' }),
      'renderSectiumShellHtml() @ packages/agent-host/sectium-shell-html.mts',
    ),
  );

  artifacts.push(
    copyOrVerifyDashboard('oversight.html', [
      path.join(MIRROR_ROOT, 'holo-v11', 'dashboard', 'oversight.html'),
      'E:\\awesome-os\\github-mirror\\awesome-os\\holo-v11\\dashboard\\oversight.html',
    ]),
  );

  artifacts.push(
    copyOrVerifyDashboard('host-manager.html', [
      path.join(MIRROR_ROOT, 'host-manager', 'console', 'index.html'),
      'E:\\awesome-os\\github-mirror\\awesome-os\\host-manager\\console\\index.html',
    ]),
  );

  const indexPath = path.join(DASHBOARDS_DIR, 'index.html');
  if (!fs.existsSync(indexPath)) {
    throw new Error(`Expected Unified Control Center at ${indexPath}`);
  }
  artifacts.push({
    name: 'index.html',
    targetPath: indexPath,
    bytes: fs.statSync(indexPath).size,
    source: 'unified-control-center',
  });

  console.log('Built AwesomeOS Agent Platform dashboards:');
  for (const item of artifacts) {
    console.log(
      `  - ${item.name.padEnd(20)} ${String(item.bytes).padStart(8)} B  [${item.source}]`,
    );
  }
}

main();
