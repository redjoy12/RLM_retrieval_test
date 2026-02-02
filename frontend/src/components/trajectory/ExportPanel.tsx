/** Export Panel Component */

import { useTrajectoryStore } from "@/stores/trajectoryStore";
import { trajectoryApi } from "@/api/client";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Download, FileJson, FileCode, FileImage } from "lucide-react";
import { downloadFile } from "@/lib/utils";

export function ExportPanel() {
  const { sessionId, trajectory } = useTrajectoryStore();

  if (!sessionId || !trajectory) {
    return null;
  }

  const handleExport = async (format: "json" | "html" | "dot") => {
    try {
      const blob = await trajectoryApi.export(sessionId, format);
      const content = await blob.text();
      const extension = format;
      downloadFile(
        content,
        `trajectory_${sessionId}.${extension}`,
        format === "json"
          ? "application/json"
          : format === "html"
          ? "text/html"
          : "text/plain"
      );
    } catch (error) {
      console.error("Export failed:", error);
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm">
          <Download className="h-4 w-4 mr-2" />
          Export
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => handleExport("json")}>
          <FileJson className="h-4 w-4 mr-2" />
          Export JSON
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => handleExport("html")}>
          <FileCode className="h-4 w-4 mr-2" />
          Export HTML Report
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => handleExport("dot")}>
          <FileImage className="h-4 w-4 mr-2" />
          Export GraphViz DOT
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
