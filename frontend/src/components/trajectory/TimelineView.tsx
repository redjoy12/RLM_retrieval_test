/** Timeline View Component */

import { useTrajectoryStore } from "@/stores/trajectoryStore";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  getStepTypeColor,
  getStepTypeLabel,
  formatDuration,
  formatTimestamp,
} from "@/lib/utils";

export function TimelineView() {
  const { trajectory } = useTrajectoryStore();

  if (!trajectory) {
    return (
      <div className="h-[500px] flex items-center justify-center text-muted-foreground">
        No timeline data available
      </div>
    );
  }

  return (
    <ScrollArea className="h-[500px]">
      <div className="space-y-2 p-4">
        {trajectory.timeline.map((event, index) => (
          <div
            key={`${event.node_id}-${index}`}
            className="flex items-center gap-4 p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
          >
            {/* Time */}
            <div className="text-xs font-mono text-muted-foreground min-w-[100px]">
              {formatTimestamp(event.start_time).split(",")[1]?.trim() ||
                event.start_time}
            </div>

            {/* Type Badge */}
            <div
              className="px-2 py-1 rounded text-xs font-medium text-white min-w-[140px] text-center"
              style={{ backgroundColor: getStepTypeColor(event.type) }}
            >
              {getStepTypeLabel(event.type)}
            </div>

            {/* Duration */}
            <div className="text-sm min-w-[80px]">
              {event.duration_ms ? formatDuration(event.duration_ms) : "—"}
            </div>

            {/* Depth */}
            <div className="text-xs text-muted-foreground">
              Depth {event.depth}
            </div>

            {/* Node ID */}
            <div className="text-xs text-muted-foreground truncate max-w-[200px]">
              {event.node_id}
            </div>
          </div>
        ))}
      </div>
    </ScrollArea>
  );
}
