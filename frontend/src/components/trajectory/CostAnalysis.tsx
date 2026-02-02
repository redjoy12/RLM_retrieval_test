/** Cost Analysis Component with Charts */

import { useTrajectoryStore } from "@/stores/trajectoryStore";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  formatCost,
  formatTokens,
  getStepTypeLabel,
  getStepTypeColor,
} from "@/lib/utils";
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Coins, BarChart3 } from "lucide-react";

export function CostAnalysis() {
  const { trajectory } = useTrajectoryStore();

  if (!trajectory) {
    return (
      <div className="h-[500px] flex items-center justify-center text-muted-foreground">
        No cost data available
      </div>
    );
  }

  const { costs, statistics } = trajectory;

  // Prepare data for charts
  const depthData = Object.entries(costs.by_depth).map(([depth, data]) => ({
    depth: `Depth ${depth}`,
    cost: data.cost_usd,
    tokens: data.total_tokens,
    count: data.count,
  }));

  const typeData = Object.entries(costs.by_type).map(([type, data]) => ({
    type: getStepTypeLabel(type as any),
    cost: data.cost_usd,
    tokens: data.total_tokens,
    color: getStepTypeColor(type as any),
  }));

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Cost</CardTitle>
            <Coins className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCost(costs.total_cost_usd)}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Tokens</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatTokens(costs.total_tokens)}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              In: {formatTokens(costs.total_input_tokens)} | Out:{" "}
              {formatTokens(costs.total_output_tokens)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">LLM Calls</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {statistics.total_llm_calls}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Max Depth</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {statistics.max_recursion_depth}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cost by Depth */}
        <Card>
          <CardHeader>
            <CardTitle>Cost by Depth</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={depthData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="depth" />
                <YAxis tickFormatter={(value) => `$${(value * 100).toFixed(0)}¢`} />
                <Tooltip formatter={(value: number) => formatCost(value)} />
                <Bar dataKey="cost" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Cost by Type */}
        <Card>
          <CardHeader>
            <CardTitle>Cost by Type</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={typeData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ type, cost }) =>
                    `${type}: ${formatCost(cost)}`
                  }
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="cost"
                >
                  {typeData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number) => formatCost(value)} />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Cost by Type Table */}
      <Card>
        <CardHeader>
          <CardTitle>Detailed Breakdown</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2">Type</th>
                  <th className="text-right py-2">Count</th>
                  <th className="text-right py-2">Input Tokens</th>
                  <th className="text-right py-2">Output Tokens</th>
                  <th className="text-right py-2">Total Tokens</th>
                  <th className="text-right py-2">Cost</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(costs.by_type).map(([type, data]) => (
                  <tr key={type} className="border-b">
                    <td className="py-2">{getStepTypeLabel(type as any)}</td>
                    <td className="text-right py-2">{data.count}</td>
                    <td className="text-right py-2">
                      {formatTokens(data.input_tokens)}
                    </td>
                    <td className="text-right py-2">
                      {formatTokens(data.output_tokens)}
                    </td>
                    <td className="text-right py-2">
                      {formatTokens(data.total_tokens)}
                    </td>
                    <td className="text-right py-2 font-medium">
                      {formatCost(data.cost_usd)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
