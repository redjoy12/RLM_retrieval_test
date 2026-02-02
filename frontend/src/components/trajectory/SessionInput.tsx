/** Session Input Component */

import { useState } from "react";
import { useTrajectoryStore } from "@/stores/trajectoryStore";
import { trajectoryApi } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Search } from "lucide-react";

export function SessionInput() {
  const [inputValue, setInputValue] = useState("");
  const { setSessionId, setTrajectory, setTreeData, setLoading, setError } =
    useTrajectoryStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const sessionId = inputValue.trim();
      setSessionId(sessionId);

      // Fetch both trajectory and tree data in parallel
      const [trajectoryData, treeData] = await Promise.all([
        trajectoryApi.getTrajectory(sessionId),
        trajectoryApi.getTree(sessionId),
      ]);

      setTrajectory(trajectoryData);
      setTreeData(treeData);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load trajectory"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardContent className="pt-6">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            placeholder="Enter session ID (e.g., session-123)"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            className="flex-1"
          />
          <Button type="submit" disabled={!inputValue.trim()}>
            <Search className="h-4 w-4 mr-2" />
            Load
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
