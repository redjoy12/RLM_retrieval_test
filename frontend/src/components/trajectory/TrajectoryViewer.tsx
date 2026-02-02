/** Main Trajectory Viewer Container */

import { useState } from "react";
import { useTrajectoryStore } from "@/stores/trajectoryStore";
import { SessionInput } from "./SessionInput";
import { TreeView } from "./TreeView";
import { TimelineView } from "./TimelineView";
import { CostAnalysis } from "./CostAnalysis";
import { CodeInspector } from "./CodeInspector";
import { NodeDetails } from "./NodeDetails";
import { ExportPanel } from "./ExportPanel";
import { LiveIndicator } from "./LiveIndicator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TreePine, Clock, DollarSign, Code } from "lucide-react";

export function TrajectoryViewer() {
  const { trajectory, isLoading, error, viewMode, setViewMode } =
    useTrajectoryStore();
  const [selectedTab, setSelectedTab] = useState(viewMode);

  const handleTabChange = (value: string) => {
    setSelectedTab(value as typeof viewMode);
    setViewMode(value as typeof viewMode);
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">
              RLM Trajectory Visualizer
            </h1>
            <p className="text-muted-foreground mt-1">
              Visualize and analyze recursive LLM execution in real-time
            </p>
          </div>
          <div className="flex items-center gap-4">
            <LiveIndicator />
            <ExportPanel />
          </div>
        </div>

        {/* Session Input */}
        <SessionInput />
      </div>

      {/* Error Display */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="pt-6">
            <p className="text-red-600">{error}</p>
          </CardContent>
        </Card>
      )}

      {/* Main Content */}
      {trajectory ? (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left Panel - Visualization */}
          <div className="lg:col-span-3">
            <Tabs value={selectedTab} onValueChange={handleTabChange}>
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="tree" className="flex items-center gap-2">
                  <TreePine className="h-4 w-4" />
                  Tree View
                </TabsTrigger>
                <TabsTrigger
                  value="timeline"
                  className="flex items-center gap-2"
                >
                  <Clock className="h-4 w-4" />
                  Timeline
                </TabsTrigger>
                <TabsTrigger value="costs" className="flex items-center gap-2">
                  <DollarSign className="h-4 w-4" />
                  Costs
                </TabsTrigger>
                <TabsTrigger value="code" className="flex items-center gap-2">
                  <Code className="h-4 w-4" />
                  Code
                </TabsTrigger>
              </TabsList>

              <TabsContent value="tree" className="mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Execution Tree</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <TreeView />
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="timeline" className="mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Execution Timeline</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <TimelineView />
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="costs" className="mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Cost Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CostAnalysis />
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="code" className="mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Code Inspection</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CodeInspector />
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>

          {/* Right Panel - Details */}
          <div className="lg:col-span-1">
            <NodeDetails />
          </div>
        </div>
      ) : (
        <Card className="p-12 text-center">
          <CardContent>
            <p className="text-muted-foreground">
              {isLoading
                ? "Loading trajectory..."
                : "Enter a session ID above to visualize a trajectory"}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
