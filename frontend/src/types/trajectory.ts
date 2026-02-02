/** Type definitions for RLM Trajectory Visualizer */

// Enums matching backend
export type TrajectoryStepType =
  | "ROOT_LLM_START"
  | "ROOT_LLM_COMPLETE"
  | "CODE_EXECUTION_START"
  | "CODE_EXECUTION_COMPLETE"
  | "SUB_LLM_SPAWN"
  | "SUB_LLM_COMPLETE"
  | "RECURSION_LIMIT_HIT"
  | "ERROR"
  | "FINAL_ANSWER";

// Cost information
export interface TokenCost {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number;
}

// Individual trajectory node
export interface TrajectoryNode {
  id: string;
  type: TrajectoryStepType;
  parent_id: string | null;
  children: string[];
  timestamp: string;
  duration_ms: number | null;
  depth: number;
  data: {
    query?: string;
    code?: string;
    output?: string;
    error?: string;
    response?: string;
    usage?: {
      input_tokens: number;
      output_tokens: number;
      total_tokens: number;
    };
    [key: string]: any;
  };
  cost: TokenCost;
}

// Complete tree structure
export interface TrajectoryTree {
  root_id: string;
  nodes: Record<string, TrajectoryNode>;
  total_nodes: number;
  max_depth: number;
  total_duration_ms: number;
  total_cost_usd: number;
  session_id: string;
}

// Timeline event
export interface TimelineEvent {
  node_id: string;
  type: TrajectoryStepType;
  start_time: string;
  end_time: string | null;
  duration_ms: number | null;
  depth: number;
  data: TrajectoryNode["data"];
}

// Cost breakdown
export interface CostBreakdown {
  by_depth: Record<
    number,
    {
      count: number;
      input_tokens: number;
      output_tokens: number;
      total_tokens: number;
      cost_usd: number;
    }
  >;
  by_type: Record<
    string,
    {
      count: number;
      input_tokens: number;
      output_tokens: number;
      total_tokens: number;
      cost_usd: number;
    }
  >;
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  total_cost_usd: number;
}

// Execution statistics
export interface ExecutionStats {
  total_steps: number;
  total_llm_calls: number;
  total_code_executions: number;
  total_errors: number;
  max_recursion_depth: number;
  total_duration_ms: number;
  avg_step_duration_ms: number;
}

// Full trajectory response
export interface TrajectoryResponse {
  session_id: string;
  tree: TrajectoryTree;
  timeline: TimelineEvent[];
  costs: CostBreakdown;
  statistics: ExecutionStats;
}

// React Flow node format
export interface FlowNode {
  id: string;
  type: "trajectoryNode" | "default";
  position: { x: number; y: number };
  data: {
    label: string;
    node_type: TrajectoryStepType;
    depth: number;
    duration_ms: number | null;
    cost: TokenCost;
    timestamp: string;
    details: TrajectoryNode["data"];
  };
  style?: React.CSSProperties;
}

// React Flow edge format
export interface FlowEdge {
  id: string;
  source: string;
  target: string;
  type: "smoothstep" | "default";
  animated?: boolean;
  style?: React.CSSProperties;
}

// Tree data in React Flow format
export interface FlowTreeData {
  session_id: string;
  root_id: string;
  nodes: FlowNode[];
  edges: FlowEdge[];
  total_nodes: number;
  max_depth: number;
  total_duration_ms: number;
  total_cost_usd: number;
}

// WebSocket message types
export type WebSocketMessageType =
  | "connection_established"
  | "step_start"
  | "step_complete"
  | "sub_llm_spawn"
  | "sub_llm_result"
  | "code_generated"
  | "code_output"
  | "error"
  | "final_result"
  | "keepalive"
  | "pong";

// WebSocket message
export interface WebSocketMessage {
  type: WebSocketMessageType;
  timestamp: string;
  session_id: string;
  data?: any;
}

// Export formats
export type ExportFormat = "json" | "html" | "dot" | "png";

// View modes
export type ViewMode = "tree" | "timeline" | "costs" | "code" | "context";

// Application state
export interface TrajectoryState {
  sessionId: string | null;
  trajectory: TrajectoryResponse | null;
  treeData: FlowTreeData | null;
  selectedNodeId: string | null;
  viewMode: ViewMode;
  isLoading: boolean;
  isLive: boolean;
  error: string | null;
  
  // Actions
  setSessionId: (id: string) => void;
  setTrajectory: (trajectory: TrajectoryResponse) => void;
  setTreeData: (treeData: FlowTreeData) => void;
  selectNode: (nodeId: string | null) => void;
  setViewMode: (mode: ViewMode) => void;
  setLoading: (loading: boolean) => void;
  setLive: (live: boolean) => void;
  setError: (error: string | null) => void;
  addTimelineEvent: (event: TimelineEvent) => void;
  clear: () => void;
}
