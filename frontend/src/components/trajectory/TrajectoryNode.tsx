/** Custom Trajectory Node for React Flow */

import { memo } from "react";
import { Handle, Position, NodeProps } from "@xyflow/react";
import { useTrajectoryStore } from "@/stores/trajectoryStore";
import {
  getStepTypeColor,
  getStepTypeLabel,
  formatDuration,
  formatTokens,
  formatCost,
} from "@/lib/utils";
import { TrajectoryNode as TrajectoryNodeType } from "@/types/trajectory";

interface TrajectoryNodeData {
  label: string;
  node_type: TrajectoryNodeType["type"];
  depth: number;
  duration_ms: number | null;
  cost: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    cost_usd: number;
  };
  timestamp: string;
  details: TrajectoryNodeType["data"];
}

export const TrajectoryNode = memo(function TrajectoryNode({
  data,
  selected,
}: NodeProps<TrajectoryNodeData>) {
  const { selectedNodeId } = useTrajectoryStore();
  const isSelected = selectedNodeId === data.id;

  const borderColor = getStepTypeColor(data.node_type);

  return (
    <div
      className={`
        relative min-w-[180px] max-w-[250px] rounded-lg border-2 
        bg-white shadow-md transition-all duration-200
        ${selected || isSelected ? "ring-2 ring-offset-2 ring-blue-500" : ""}
        hover:shadow-lg
      `}
      style={{ borderColor }}
    >
      {/* Target Handle */}
      <Handle
        type="target"
        position={Position.Left}
        className="!w-3 !h-3 !bg-slate-400"
      />

      {/* Content */}
      <div className="p-3 space-y-2">
        {/* Header */}
        <div className="font-semibold text-sm leading-tight">
          {getStepTypeLabel(data.node_type)}
        </div>

        {/* Metadata */}
        <div className="text-xs text-muted-foreground space-y-1">
          {data.duration_ms !== null && (
            <div className="flex items-center gap-1">
              <span>⏱️</span>
              <span>{formatDuration(data.duration_ms)}</span>
            </div>
          )}

          {data.cost.total_tokens > 0 && (
            <>
              <div className="flex items-center gap-1">
                <span>🪙</span>
                <span>{formatTokens(data.cost.total_tokens)} tokens</span>
              </div>
              <div className="flex items-center gap-1">
                <span>💰</span>
                <span>{formatCost(data.cost.cost_usd)}</span>
              </div>
            </>
          )}

          <div className="flex items-center gap-1">
            <span>📊</span>
            <span>Depth {data.depth}</span>
          </div>
        </div>
      </div>

      {/* Source Handle */}
      <Handle
        type="source"
        position={Position.Right}
        className="!w-3 !h-3 !bg-slate-400"
      />
    </div>
  );
});
