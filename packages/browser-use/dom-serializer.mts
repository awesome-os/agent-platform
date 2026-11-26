/**
 * @file packages/browser-use/dom-serializer.mts
 * @description Indexed DOM tree serializer (`DOMTreeSerializer`) for autonomous browser
 * agents. Produces compact `[idx] <tag role="...">"text"` maps with visibility filtering
 * (hidden/display:none/visibility:hidden/opacity:0/zero-bounds) and bounded token budgets.
 *
 * Strictly `erasableSyntaxOnly: true`, zero external runtime dependencies.
 */

export interface DomRect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface DomNodeSnapshot {
  readonly tag: string;
  readonly role?: string;
  readonly text?: string;
  readonly ariaLabel?: string;
  readonly placeholder?: string;
  readonly name?: string;
  readonly id?: string;
  readonly type?: string;
  readonly href?: string;
  readonly value?: string;
  readonly selector?: string;
  readonly backendNodeId?: number;
  readonly tabIndex?: number;
  readonly checked?: boolean;
  readonly disabled?: boolean;
  readonly hidden?: boolean;
  readonly ariaHidden?: boolean;
  readonly visible?: boolean;
  readonly interactive?: boolean;
  readonly contentEditable?: boolean;
  readonly style?: string;
  readonly bounds?: DomRect;
  readonly children?: readonly DomNodeSnapshot[];
}

export interface DomSerializerOptions {
  readonly maxTokens?: number;
  readonly maxElements?: number;
  readonly charsPerToken?: number;
  readonly startIndex?: number;
  readonly includeDisabled?: boolean;
  readonly maxTextLength?: number;
  readonly viewport?: { readonly width: number; readonly height: number };
}

export interface IndexedDomElement {
  readonly index: number;
  readonly tag: string;
  readonly role: string;
  readonly text: string;
  readonly selector: string;
  readonly backendNodeId?: number;
  readonly href?: string;
  readonly type?: string;
  readonly name?: string;
  readonly value?: string;
  readonly line: string;
}

export interface SerializedDomMap {
  readonly text: string;
  readonly lines: readonly string[];
  readonly elements: readonly IndexedDomElement[];
  readonly selectorMap: ReadonlyMap<number, IndexedDomElement>;
  readonly estimatedTokens: number;
  readonly maxTokens: number;
  readonly truncated: boolean;
  readonly totalInteractiveVisible: number;
  readonly omittedCount: number;
}

const INTERACTIVE_TAGS = new Set([
  'a',
  'button',
  'input',
  'select',
  'textarea',
  'option',
  'summary',
  'details',
  'menuitem',
]);

const INTERACTIVE_ROLES = new Set([
  'button',
  'link',
  'textbox',
  'searchbox',
  'checkbox',
  'radio',
  'combobox',
  'listbox',
  'option',
  'menuitem',
  'menuitemcheckbox',
  'menuitemradio',
  'tab',
  'switch',
  'slider',
  'spinbutton',
]);

const DEFAULT_TAG_ROLES: Readonly<Record<string, string>> = {
  a: 'link',
  button: 'button',
  select: 'combobox',
  textarea: 'textbox',
  option: 'option',
  summary: 'button',
  details: 'group',
};

function inferRole(node: DomNodeSnapshot, tag: string): string {
  if (node.role && node.role.trim().length > 0) {
    return node.role.trim().toLowerCase();
  }
  if (tag === 'input') {
    const t = (node.type ?? 'text').toLowerCase();
    if (t === 'checkbox') return 'checkbox';
    if (t === 'radio') return 'radio';
    if (t === 'submit' || t === 'button' || t === 'reset') return 'button';
    if (t === 'search') return 'searchbox';
    if (t === 'range') return 'slider';
    return 'textbox';
  }
  return DEFAULT_TAG_ROLES[tag] ?? (node.contentEditable ? 'textbox' : 'generic');
}

function hasInvisibleStyle(style?: string): boolean {
  if (!style) {
    return false;
  }
  const normalized = style.toLowerCase().replace(/\s+/g, '');
  return (
    normalized.includes('display:none') ||
    normalized.includes('visibility:hidden') ||
    normalized.includes('visibility:collapse') ||
    /(?:^|;)opacity:0(?:\.0+)?(?:;|$)/.test(normalized)
  );
}

function isOutsideViewport(
  bounds: DomRect | undefined,
  viewport: { readonly width: number; readonly height: number } | undefined,
): boolean {
  if (!bounds || !viewport) {
    return false;
  }
  return (
    bounds.x + bounds.width <= 0 ||
    bounds.y + bounds.height <= 0 ||
    bounds.x >= viewport.width ||
    bounds.y >= viewport.height
  );
}

export function isDomNodeVisible(
  node: DomNodeSnapshot,
  viewport?: { readonly width: number; readonly height: number },
): boolean {
  if (node.visible === false || node.hidden === true || node.ariaHidden === true) {
    return false;
  }
  if (hasInvisibleStyle(node.style)) {
    return false;
  }
  if (node.bounds && (node.bounds.width <= 0 || node.bounds.height <= 0)) {
    return false;
  }
  if (isOutsideViewport(node.bounds, viewport)) {
    return false;
  }
  return true;
}

export function isDomNodeInteractive(node: DomNodeSnapshot): boolean {
  if (typeof node.interactive === 'boolean') {
    return node.interactive;
  }
  const tag = node.tag.trim().toLowerCase();
  if (tag === 'input' && (node.type ?? '').toLowerCase() === 'hidden') {
    return false;
  }
  if (INTERACTIVE_TAGS.has(tag)) {
    return true;
  }
  if (node.role && INTERACTIVE_ROLES.has(node.role.trim().toLowerCase())) {
    return true;
  }
  if (node.contentEditable === true) {
    return true;
  }
  if (typeof node.tabIndex === 'number' && node.tabIndex >= 0) {
    return true;
  }
  return false;
}

function normalizeText(raw: string | undefined, maxLen: number): string {
  if (!raw) {
    return '';
  }
  const collapsed = raw.replace(/\s+/g, ' ').trim();
  if (collapsed.length <= maxLen) {
    return collapsed;
  }
  return `${collapsed.slice(0, Math.max(1, maxLen - 3))}...`;
}

function escapeQuotes(text: string): string {
  return text.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

function buildSelector(node: DomNodeSnapshot, tag: string, ordinal: number): string {
  if (node.selector && node.selector.trim().length > 0) {
    return node.selector.trim();
  }
  if (node.id && node.id.trim().length > 0) {
    return `#${node.id.trim()}`;
  }
  if (node.name && node.name.trim().length > 0) {
    return `${tag}[name="${node.name.trim()}"]`;
  }
  if (node.ariaLabel && node.ariaLabel.trim().length > 0) {
    return `${tag}[aria-label="${node.ariaLabel.trim()}"]`;
  }
  return `${tag}:nth-of-type(${ordinal})`;
}

/**
 * Lightweight zero-dependency HTML snippet parser producing `DomNodeSnapshot` trees
 * for deterministic tests and HTML transport payloads.
 */
export function parseHtmlToDomSnapshot(html: string): DomNodeSnapshot {
  const root: { tag: string; children: DomNodeSnapshot[] } = { tag: 'document', children: [] };
  const stack: Array<{ tag: string; attrs: Record<string, string>; textParts: string[]; children: DomNodeSnapshot[] }> = [];
  const voidTags = new Set(['input', 'img', 'br', 'hr', 'meta', 'link', 'source']);
  const tokenRe = /<(\/?)([a-zA-Z0-9-]+)([^>]*)>|([^<]+)/g;

  const parseAttrs = (raw: string): Record<string, string> => {
    const attrs: Record<string, string> = {};
    const attrRe = /([a-zA-Z_:][-a-zA-Z0-9_:.]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+)))?/g;
    let m: RegExpExecArray | null;
    while ((m = attrRe.exec(raw)) !== null) {
      const k = m[1]!.toLowerCase();
      const v = m[2] ?? m[3] ?? m[4] ?? '';
      attrs[k] = v;
    }
    return attrs;
  };

  const toSnapshot = (frame: {
    tag: string;
    attrs: Record<string, string>;
    textParts: string[];
    children: DomNodeSnapshot[];
  }): DomNodeSnapshot => {
    const a = frame.attrs;
    const text = frame.textParts.join(' ').replace(/\s+/g, ' ').trim();
    return {
      tag: frame.tag,
      ...(a.role !== undefined ? { role: a.role } : {}),
      ...(text ? { text } : {}),
      ...(a['aria-label'] !== undefined ? { ariaLabel: a['aria-label'] } : {}),
      ...(a.placeholder !== undefined ? { placeholder: a.placeholder } : {}),
      ...(a.name !== undefined ? { name: a.name } : {}),
      ...(a.id !== undefined ? { id: a.id } : {}),
      ...(a.type !== undefined ? { type: a.type } : {}),
      ...(a.href !== undefined ? { href: a.href } : {}),
      ...(a.value !== undefined ? { value: a.value } : {}),
      ...(a.style !== undefined ? { style: a.style } : {}),
      ...('hidden' in a ? { hidden: true } : {}),
      ...(a['aria-hidden'] === 'true' ? { ariaHidden: true } : {}),
      ...('disabled' in a ? { disabled: true } : {}),
      ...(a.tabindex !== undefined && !Number.isNaN(Number(a.tabindex))
        ? { tabIndex: Number(a.tabindex) }
        : {}),
      ...(frame.children.length > 0 ? { children: frame.children } : {}),
    };
  };

  let match: RegExpExecArray | null;
  while ((match = tokenRe.exec(html)) !== null) {
    if (match[4] !== undefined) {
      const txt = match[4].replace(/\s+/g, ' ').trim();
      if (txt && stack.length > 0) {
        stack[stack.length - 1]!.textParts.push(txt);
      }
      continue;
    }
    const isClosing = match[1] === '/';
    const tag = match[2]!.toLowerCase();
    const rawAttrs = match[3] ?? '';
    const selfClosing = rawAttrs.trim().endsWith('/') || voidTags.has(tag);

    if (isClosing) {
      for (let i = stack.length - 1; i >= 0; i -= 1) {
        if (stack[i]!.tag === tag) {
          while (stack.length > i) {
            const closed = stack.pop()!;
            const snap = toSnapshot(closed);
            if (stack.length > 0) {
              stack[stack.length - 1]!.children.push(snap);
            } else {
              root.children.push(snap);
            }
          }
          break;
        }
      }
      continue;
    }

    const frame = {
      tag,
      attrs: parseAttrs(rawAttrs),
      textParts: [] as string[],
      children: [] as DomNodeSnapshot[],
    };
    if (selfClosing) {
      const snap = toSnapshot(frame);
      if (stack.length > 0) {
        stack[stack.length - 1]!.children.push(snap);
      } else {
        root.children.push(snap);
      }
    } else {
      stack.push(frame);
    }
  }

  while (stack.length > 0) {
    const closed = stack.pop()!;
    const snap = toSnapshot(closed);
    if (stack.length > 0) {
      stack[stack.length - 1]!.children.push(snap);
    } else {
      root.children.push(snap);
    }
  }

  return {
    tag: 'document',
    children: root.children,
  };
}

/**
 * Serializes a DOM snapshot tree (or raw HTML string) into indexed interactive
 * element lines `[idx] <tag role="...">"text"` under visibility and token constraints.
 */
export class DOMTreeSerializer {
  private readonly defaultMaxTokens: number;
  private readonly defaultMaxElements: number;
  private readonly charsPerToken: number;
  private readonly startIndex: number;
  private readonly includeDisabled: boolean;
  private readonly maxTextLength: number;

  constructor(options: DomSerializerOptions = {}) {
    this.defaultMaxTokens = Math.max(16, options.maxTokens ?? 2000);
    this.defaultMaxElements = Math.max(1, options.maxElements ?? 250);
    this.charsPerToken = Math.max(1, options.charsPerToken ?? 4);
    this.startIndex = options.startIndex ?? 1;
    this.includeDisabled = options.includeDisabled ?? false;
    this.maxTextLength = Math.max(16, options.maxTextLength ?? 120);
  }

  public serialize(
    input: DomNodeSnapshot | string,
    overrides?: DomSerializerOptions,
  ): SerializedDomMap {
    const tree = typeof input === 'string' ? parseHtmlToDomSnapshot(input) : input;
    const maxTokens = Math.max(16, overrides?.maxTokens ?? this.defaultMaxTokens);
    const maxElements = Math.max(1, overrides?.maxElements ?? this.defaultMaxElements);
    const charsPerToken = Math.max(1, overrides?.charsPerToken ?? this.charsPerToken);
    const startIndex = overrides?.startIndex ?? this.startIndex;
    const includeDisabled = overrides?.includeDisabled ?? this.includeDisabled;
    const maxTextLength = Math.max(16, overrides?.maxTextLength ?? this.maxTextLength);
    const viewport = overrides?.viewport;

    const candidates: Array<{
      readonly node: DomNodeSnapshot;
      readonly tag: string;
      readonly role: string;
      readonly text: string;
      readonly selector: string;
    }> = [];

    let tagCounter = 0;

    const walk = (node: DomNodeSnapshot, ancestorHidden: boolean): void => {
      const visibleHere = !ancestorHidden && isDomNodeVisible(node, viewport);
      if (!visibleHere) {
        return;
      }
      const tag = node.tag.trim().toLowerCase();
      const interactive = isDomNodeInteractive(node);
      if (interactive && (includeDisabled || node.disabled !== true)) {
        tagCounter += 1;
        const role = inferRole(node, tag);
        const rawText =
          node.text ??
          node.ariaLabel ??
          node.placeholder ??
          node.value ??
          node.name ??
          '';
        const text = normalizeText(rawText, maxTextLength);
        const selector = buildSelector(node, tag, tagCounter);
        candidates.push({ node, tag, role, text, selector });
      }
      if (node.children) {
        for (const child of node.children) {
          walk(child, false);
        }
      }
    };

    walk(tree, false);

    const elements: IndexedDomElement[] = [];
    const lines: string[] = [];
    const selectorMap = new Map<number, IndexedDomElement>();
    let estimatedTokens = 0;
    let truncated = false;

    for (let i = 0; i < candidates.length; i += 1) {
      if (elements.length >= maxElements) {
        truncated = true;
        break;
      }
      const c = candidates[i]!;
      const index = startIndex + elements.length;
      const extraParts: string[] = [];
      if (c.node.type) extraParts.push(`type="${escapeQuotes(c.node.type)}"`);
      if (c.node.name) extraParts.push(`name="${escapeQuotes(c.node.name)}"`);
      if (c.node.href) extraParts.push(`href="${escapeQuotes(normalizeText(c.node.href, 64))}"`);
      if (c.node.placeholder && c.text !== normalizeText(c.node.placeholder, maxTextLength)) {
        extraParts.push(`placeholder="${escapeQuotes(normalizeText(c.node.placeholder, 48))}"`);
      }
      if (c.node.disabled) extraParts.push('disabled="true"');
      const attrSuffix = extraParts.length > 0 ? ` ${extraParts.join(' ')}` : '';
      const line = `[${index}] <${c.tag} role="${escapeQuotes(c.role)}"${attrSuffix}>"${escapeQuotes(c.text)}"`;
      const lineTokens = Math.max(1, Math.ceil((line.length + 1) / charsPerToken));

      if (estimatedTokens + lineTokens > maxTokens) {
        truncated = true;
        break;
      }

      estimatedTokens += lineTokens;
      const entry: IndexedDomElement = Object.freeze({
        index,
        tag: c.tag,
        role: c.role,
        text: c.text,
        selector: c.selector,
        ...(c.node.backendNodeId !== undefined ? { backendNodeId: c.node.backendNodeId } : {}),
        ...(c.node.href !== undefined ? { href: c.node.href } : {}),
        ...(c.node.type !== undefined ? { type: c.node.type } : {}),
        ...(c.node.name !== undefined ? { name: c.node.name } : {}),
        ...(c.node.value !== undefined ? { value: c.node.value } : {}),
        line,
      });
      elements.push(entry);
      lines.push(line);
      selectorMap.set(index, entry);
    }

    return Object.freeze({
      text: lines.join('\n'),
      lines: Object.freeze(lines),
      elements: Object.freeze(elements),
      selectorMap,
      estimatedTokens,
      maxTokens,
      truncated,
      totalInteractiveVisible: candidates.length,
      omittedCount: Math.max(0, candidates.length - elements.length),
    });
  }
}
