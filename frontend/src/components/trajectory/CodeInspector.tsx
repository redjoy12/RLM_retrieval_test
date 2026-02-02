/** Code Inspector Component */

import { useTrajectoryStore } from "@/stores/trajectoryStore";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Copy, Download } from "lucide-react";
import { Highlight, themes } from "prism-react-renderer";
import { downloadFile } from "@/lib/utils";

export function CodeInspector() {
  const { trajectory, selectedNodeId } = useTrajectoryStore();

  if (!trajectory) {
    return (
      <div className="h-[500px] flex items-center justify-center text-muted-foreground">
        No code data available
      </div>
    );
  }

  // Get code from selected node or find first code node
  let code = "";
  let output = "";
  let language = "python";

  if (selectedNodeId) {
    const node = trajectory.tree.nodes[selectedNodeId];
    if (node) {
      code = node.data.code || "";
      output = node.data.output || "";
    }
  }

  // If no code in selected node, find first code execution
  if (!code) {
    const codeNode = Object.values(trajectory.tree.nodes).find(
      (n) => n.data.code
    );
    if (codeNode) {
      code = codeNode.data.code || "";
      output = codeNode.data.output || "";
    }
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
  };

  const handleDownload = () => {
    downloadFile(code, `trajectory_${trajectory.session_id}_code.py`, "text/x-python");
  };

  return (
    <div className="space-y-4">
      {/* Generated Code */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Generated Code</CardTitle>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={handleCopy}>
              <Copy className="h-4 w-4 mr-2" />
              Copy
            </Button>
            <Button variant="outline" size="sm" onClick={handleDownload}>
              <Download className="h-4 w-4 mr-2" />
              Download
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[300px]">
            {code ? (
              <Highlight theme={themes.vsDark} code={code} language={language}>
                {({ className, style, tokens, getLineProps, getTokenProps }) => (
                  <pre
                    className={`${className} text-sm font-mono p-4 rounded`}
                    style={style}
                  >
                    {tokens.map((line, i) => (
                      <div key={i} {...getLineProps({ line })}>
                        <span className="inline-block w-8 text-gray-500 select-none">
                          {i + 1}
                        </span>
                        {line.map((token, key) => (
                          <span key={key} {...getTokenProps({ token })} />
                        ))}
                      </div>
                    ))}
                  </pre>
                )}
              </Highlight>
            ) : (
              <p className="text-muted-foreground">No code available</p>
            )}
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Output */}
      {output && (
        <Card>
          <CardHeader>
            <CardTitle>Execution Output</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[200px]">
              <pre className="text-sm font-mono bg-muted p-4 rounded whitespace-pre-wrap">
                {output}
              </pre>
            </ScrollArea>
          </CardContent>
        </Card>
      )}

      {/* All Code Blocks */}
      <Card>
        <CardHeader>
          <CardTitle>All Code Blocks</CardTitle>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[200px]">
            <div className="space-y-2">
              {Object.values(trajectory.tree.nodes)
                .filter((n) => n.data.code)
                .map((node) => (
                  <div
                    key={node.id}
                    className="p-3 rounded-lg bg-muted cursor-pointer hover:bg-muted/80 transition-colors"
                    onClick={() => {
                      // Would select this node
                    }}
                  >
                    <p className="text-sm font-medium truncate">
                      {node.data.code?.split("\n")[0] || "Code block"}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {node.id} • {node.data.code?.split("\n").length || 0} lines
                    </p>
                  </div>
                ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}
