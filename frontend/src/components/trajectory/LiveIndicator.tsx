/** Live Indicator Component */

import { useTrajectoryStore } from "@/stores/trajectoryStore";
import { Badge } from "@/components/ui/badge";
import { useWebSocket } from "@/hooks/useWebSocket";

export function LiveIndicator() {
  const { sessionId } = useTrajectoryStore();
  const { isConnected } = useWebSocket(sessionId);

  if (!sessionId) {
    return null;
  }

  return (
    <div className="flex items-center gap-2">
      {isConnected ? (
        <Badge
          variant="outline"
          className="bg-green-50 text-green-700 border-green-200 animate-pulse"
        >
          <span className="w-2 h-2 bg-green-500 rounded-full mr-2" />
          Live
        </Badge>
      ) : (
        <Badge variant="outline" className="text-muted-foreground">
          <span className="w-2 h-2 bg-gray-300 rounded-full mr-2" />
          Offline
        </Badge>
      )}
    </div>
  );
}
