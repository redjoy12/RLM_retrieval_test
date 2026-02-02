/** Tree View Component using React Flow */

import { useCallback, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Panel,
  Node,
  Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useTrajectoryStore } from "@/stores/trajectoryStore";
import { TrajectoryNode } from "./TrajectoryNode";
import { Button } from "@/components/ui/button";
import { Maximize2, Layout } from "lucide-react";

const nodeTypes = {
  trajectoryNode: TrajectoryNode,
};

export function TreeView() {
  const { treeData, selectNode } = useTrajectoryStore();

  // Initialize nodes and edges from tree data
  const initialNodes: Node[] = useMemo(() => {
    if (!treeData) return [];
    return treeData.nodes.map((node) => ({
      ...node,
      // Calculate hierarchical position
      position: calculateNodePosition(node.data.depth, node.id, treeData),
    }));
  }, [treeData]);

  const initialEdges: Edge[] = useMemo(() => {
    if (!treeData) return [];
    return treeData.edges;
  }, [treeData]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Handle node click
  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      selectNode(node.id);
    },
    [selectNode]
  );

  // Auto-layout function
  const autoLayout = useCallback(() => {
    if (!treeData) return;

    const updatedNodes = nodes.map((node) => ({
      ...node,
      position: calculateNodePosition(node.data.depth, node.id, treeData),
    }));

    setNodes(updatedNodes);
  }, [nodes, setNodes, treeData]);

  // Fit view
  const fitView = useCallback(() => {
    // This would be called on the ReactFlow instance
    // We'll need to use a ref for this
  }, []);

  if (!treeData) {
    return (
      <div className="h-[500px] flex items-center justify-center text-muted-foreground">
        No tree data available
      </div>
    );
  }

  return (
    <div className="h-[600px] border rounded-lg">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        attributionPosition="bottom-left"
      >
        <Background color="#f1f5f9" gap={16} />
        <Controls />
        <MiniMap
          nodeColor={(node) => node.style?.background?.toString() || "#94a3b8"}
          className="bg-white border rounded-lg shadow-lg"
        />
        <Panel position="top-right" className="flex gap-2">
          <Button variant="outline" size="sm" onClick={autoLayout}>
            <Layout className="h-4 w-4 mr-2" />
            Auto Layout
          </Button>
          <Button variant="outline" size="sm">
            <Maximize2 className="h-4 w-4 mr-2" />
            Fit View
          </Button>
        </Panel>
      </ReactFlow>
    </div>
  );
}

/** Calculate node position in hierarchical layout */
function calculateNodePosition(
  depth: number,
  nodeId: string,
  treeData: { nodes: { id: string; data: { depth: number } }[] }
): { x: number; y: number } {
  const xSpacing = 300;
  const ySpacing = 100;

  // Count nodes at this depth before this node
  const nodesAtDepth = treeData.nodes.filter((n) => n.data.depth === depth);
  const indexAtDepth = nodesAtDepth.findIndex((n) => n.id === nodeId);

  return {
    x: depth * xSpacing,
    y: indexAtDepth * ySpacing,
  };
}
