/** Transport- and host-independent PES preview data structures. */
export interface SourceLocation { line: number; column: number }
export interface SourceRange { start: SourceLocation; end: SourceLocation }
export interface Diagnostic {
  message: string;
  severity: 'error' | 'warning';
  range: SourceRange;
}
export interface ParameterBlock {
  /** Unique within a view, including the component / group path. */
  id: string;
  name: string;
  type: string;
  /** Unmodified source, including indentation and comments. */
  source: string;
  range: SourceRange;
  parent?: string;
  children: string[];
}
export interface GraphNode {
  id: string;
  name: string;
  type?: string;
  controllerType?: string;
  firstLine: number;
  source: SourceRange;
  ports: string[];
  parameterBlock?: string;
}
export interface GraphEdge {
  id: string;
  from: string;
  to: string;
  fromPort?: string;
  toPort?: string;
  directed: boolean;
  line: number;
  source: SourceRange;
}
export interface GraphGroup {
  id: string;
  name: string;
  /** Node or group IDs, groups use an & prefix. */
  members: string[];
  line: number;
  source: SourceRange;
  parent?: string;
}
export interface GraphView {
  name: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  groups: GraphGroup[];
  parameters: ParameterBlock[];
  source: SourceRange;
}
export interface GraphEvent {
  name: string;
  type: 'connect' | 'disconnect';
  target: string;
  time: number;
  line: number;
}
export interface EventSegment {
  index: number;
  start: number;
  end: number;
  /** A list of deactivated node and/or edge labels after events at start. */
  disconnected: string[];
}
export interface PESModel {
  views: GraphView[];
  globals: ParameterBlock[];
  events: GraphEvent[];
  timeline: EventSegment[];
  diagnostics: Diagnostic[];
  /** Name of initially displayed view, usually circuit. */
  defaultView: string;
}
export interface LayoutNode { id: string; x: number; y: number; width: number; height: number }
export interface LayoutGroup { id: string; x: number; y: number; width: number; height: number; depth: number }
export interface GraphLayout {
  nodes: Map<string, LayoutNode>;
  groups: Map<string, LayoutGroup>;
  left: number;
  top: number;
  width: number;
  height: number;
}
export interface NodeOffset { dx: number; dy: number }
export interface PreviewLayout {
  format: 'peslite.preview-layout';
  version: 1;
  views: Record<string, Record<string, NodeOffset>>;
}
