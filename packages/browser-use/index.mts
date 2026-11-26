/**
 * @file packages/browser-use/index.mts
 * @description Barrel exports for `@awesome-os/agent-platform` `browser-use`
 * (Holo v11 + `sectium-device` / CDP pipe).
 *
 * Strictly `erasableSyntaxOnly: true`.
 */

export {
  DOMTreeSerializer,
  isDomNodeInteractive,
  isDomNodeVisible,
  parseHtmlToDomSnapshot,
  type DomNodeSnapshot,
  type DomRect,
  type DomSerializerOptions,
  type IndexedDomElement,
  type SerializedDomMap,
} from './dom-serializer.mts';

export {
  BrowserUseController,
  MockBrowserTransport,
  SectiumDeviceOfflineError,
  SectiumPipeTransport,
  getDefaultSectiumPipePath,
  type BrowserActionName,
  type BrowserActionResult,
  type BrowserCommandMethod,
  type BrowserObservation,
  type BrowserTransport,
  type BrowserTransportRequest,
  type BrowserTransportResponse,
  type MockPageState,
  type RecordedTransportCall,
  type ScreenshotCapture,
  type SectiumPipeTransportOptions,
} from './controller.mts';

export {
  BrowserUseAgent,
  type BrowserAgentDecision,
  type BrowserHistoryStep,
  type BrowserPlannerContext,
  type BrowserUseAgentOptions,
  type BrowserUsePlanner,
  type BrowserUseReceipt,
  type BrowserUseRunResult,
} from './agent.mts';
