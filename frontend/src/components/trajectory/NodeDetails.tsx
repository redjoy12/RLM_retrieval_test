/** Node Details Panel */

import { useTrajectoryStore } from "@/stores/trajectoryStore";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  getStepTypeLabel,
  getStepTypeColor,
  formatDuration,
  formatTokens,
  formatCost,
  formatTimestamp,
} from "@/lib/utils";
import { X, Clock, Coins, Database, Code, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";

export function NodeDetails() {
  const { trajectory, selectedNodeId, selectNode } = useTrajectoryStore();

  if (!trajectory || !selectedNodeId) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Node Details</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Select a node from the tree to view details
          </p>
        </CardContent>
      </Card>
    );
  }

  const node = trajectory.tree.nodes[selectedNodeId];
  if (!node) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Node Details</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Node not found</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-fit">
      <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-2">
        <CardTitle className="text-lg">Node Details</CardTitle>
        <Button variant="ghost" size="sm" onClick={() => selectNode(null)}>
          <X className="h-4 w-4" />
        </Button>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[calc(100vh-400px)]">
          <div className="space-y-4">
            {/* Type Badge */}
            <div>
              <Badge
                style={{
                  backgroundColor: getStepTypeColor(node.type),
                  color: "white",
                }}
              >
                {getStepTypeLabel(node.type)}
              </Badge>
            </div>

            {/* ID */}
            <div className="text-xs text-muted-foreground break-all">
              ID: {node.id}
            </div>

            {/* Basic Info */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm">
                <Clock className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Duration:</span>
                <span className="font-medium">
                  {formatDuration(node.duration_ms)}
                </span>
              </div>

              <div className="flex items-center gap-2 text-sm">
                <Database className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Depth:</span>
                <span className="font-medium">{node.depth}</span>
              </div>

              <div className="flex items-center gap-2 text-sm">
                <Clock className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Timestamp:</span>
                <span className="font-medium">
                  {formatTimestamp(node.timestamp)}
                </span>
              </div>
            </div>

            {/* Cost Info */}
            {node.cost.total_tokens > 0 && (
              <div className="border-t pt-4">
                <div className="flex items-center gap-2 mb-2">
                  <Coins className="h-4 w-4 text-muted-foreground" />
                  <span className="font-medium">Cost Information</span>
                </div>
                <div className="space-y-1 text-sm pl-6">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Input tokens:</span>
                    <span>{formatTokens(node.cost.input_tokens)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">
                      Output tokens:
                    </span>
                    <span>{formatTokens(node.cost.output_tokens)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Total tokens:</span>
                    <span>{formatTokens(node.cost.total_tokens)}</span>
                  </div>
                  <div className="flex justify-between font-medium">
                    <span className="text-muted-foreground">Cost:</span>
                    <span>{formatCost(node.cost.cost_usd)}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Query */}
            {node.data.query && (
              <div className="border-t pt-4">
                <div className="flex items-center gap-2 mb-2">
                  <MessageSquare className="h-4 w-4 text-muted-foreground" />
                  <span className="font-medium">Query</span>
                </div>
                <p className="text-sm pl-6 whitespace-pre-wrap">
                  {node.data.query}
                </p>
              </div>
            )}

            {/* Response */}
            {node.data.response && (
              <div className="border-t pt-4">
                <div className="flex items-center gap-2 mb-2">
                  <MessageSquare className="h-4 w-4 text-muted-foreground" />
                  <span className="font-medium">Response</span>
                </div>
                <p className="text-sm pl-6 whitespace-pre-wrap">
                  {node.data.response}
                </p>
              </div>
            )}

            {/* Code */}
            {node.data.code && (
              <div className="border-t pt-4">
                <div className="flex items-center gap-2 mb-2">
                  <Code className="h-4 w-4 text-muted-foreground" />
                  <span className="font-medium">Generated Code</span>
                </div>
                <pre className="text-xs bg-muted p-2 rounded overflow-x-auto">
                  {node.data.code}
                </pre>
              </div>
            )}

            {/* Output */}
            {node.data.output && (
              <div className="border-t pt-4">
                <div className="flex items-center gap-2 mb-2">
                  <Code className="h-4 w-4 text-muted-foreground" />
                  <span className="font-medium">Output</span>
                </div>
                <pre className="text-xs bg-muted p-2 rounded overflow-x-auto">
                  {node.data.output}
                </pre>
              </div>
            )}

            {/* Error */}
            {node.data.error && (
              <div className="border-t pt-4">
                <div className="flex items-center gap-2 mb-2 text-red-500">
                  <X className="h-4 w-4" />
                  <span className="font-medium">Error</span>
                </div>
                <p className="text-sm pl-6 text-red-600 whitespace-pre-wrap">
                  {node.data.error}
                </p>
              </div>
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
