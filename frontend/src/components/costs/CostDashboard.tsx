/** Cost Dashboard Component with Charts */

import { useState } from "react";
import { DollarSign, TrendingUp, Activity, CreditCard } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
} from "recharts";

// Mock data - would come from API
const costByDay = [
  { date: "2026-01-25", cost: 2.5 },
  { date: "2026-01-26", cost: 3.2 },
  { date: "2026-01-27", cost: 1.8 },
  { date: "2026-01-28", cost: 4.1 },
  { date: "2026-01-29", cost: 2.9 },
  { date: "2026-01-30", cost: 3.5 },
  { date: "2026-01-31", cost: 2.2 },
];

const costByModel = [
  { name: "GPT-5 Mini", cost: 12.5, color: "#3b82f6" },
  { name: "GPT-5", cost: 8.3, color: "#8b5cf6" },
  { name: "Claude Sonnet", cost: 5.2, color: "#22c55e" },
  { name: "Claude Opus", cost: 3.1, color: "#f59e0b" },
];

const tokenUsage = [
  { time: "00:00", input: 1200, output: 800 },
  { time: "04:00", input: 800, output: 500 },
  { time: "08:00", input: 2500, output: 1800 },
  { time: "12:00", input: 3200, output: 2400 },
  { time: "16:00", input: 2800, output: 2100 },
  { time: "20:00", input: 1900, output: 1400 },
  { time: "23:59", input: 1100, output: 700 },
];

const timeRanges = [
  { value: "7d", label: "Last 7 days" },
  { value: "30d", label: "Last 30 days" },
  { value: "90d", label: "Last 90 days" },
] as const;

export function CostDashboard() {
  const [timeRange, setTimeRange] = useState<"7d" | "30d" | "90d">("7d");

  const totalCost = costByDay.reduce((sum, day) => sum + day.cost, 0);
  const totalQueries = 156;
  const avgCostPerQuery = totalCost / totalQueries;
  const budgetLimit = 50.0;
  const budgetUsedPercent = (totalCost / budgetLimit) * 100;

  const stats = [
    {
      title: "Total Cost",
      value: `$${totalCost.toFixed(2)}`,
      icon: DollarSign,
      description: `of $${budgetLimit} budget`,
    },
    {
      title: "Total Queries",
      value: totalQueries.toString(),
      icon: Activity,
      description: `${avgCostPerQuery.toFixed(3)} USD avg`,
    },
    {
      title: "Budget Used",
      value: `${budgetUsedPercent.toFixed(1)}%`,
      icon: CreditCard,
      description: budgetUsedPercent > 80 ? "Approaching limit!" : "On track",
    },
    {
      title: "Daily Trend",
      value: "+12%",
      icon: TrendingUp,
      description: "vs last week",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <Card key={stat.title} variant="elevated">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
              <stat.icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className="text-xs text-muted-foreground">{stat.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Time Range Selector */}
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium">Time Range:</span>
        <Select
          value={timeRange}
          onValueChange={(value) => setTimeRange(value as typeof timeRange)}
        >
          <SelectTrigger className="w-[150px] h-9 shadow-elevation-sm">
            <SelectValue placeholder="Select time range" />
          </SelectTrigger>
          <SelectContent>
            {timeRanges.map((range) => (
              <SelectItem key={range.value} value={range.value}>
                {range.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cost by Day */}
        <Card variant="elevated">
          <CardHeader className="bg-gradient-to-r from-card to-sidebar/30">
            <CardTitle>Daily Costs</CardTitle>
          </CardHeader>
          <CardContent className="pt-6">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={costByDay}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  tickFormatter={(date) =>
                    new Date(date).toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric",
                    })
                  }
                />
                <YAxis tickFormatter={(value) => `$${value}`} />
                <Tooltip
                  formatter={(value: number) => [`$${value.toFixed(2)}`, "Cost"]}
                  labelFormatter={(label) =>
                    new Date(label).toLocaleDateString("en-US", {
                      weekday: "long",
                      year: "numeric",
                      month: "long",
                      day: "numeric",
                    })
                  }
                />
                <Bar dataKey="cost" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Cost by Model */}
        <Card variant="elevated">
          <CardHeader className="bg-gradient-to-r from-card to-sidebar/30">
            <CardTitle>Cost by Model</CardTitle>
          </CardHeader>
          <CardContent className="pt-6">
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={costByModel}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) =>
                    `${name}: ${(percent * 100).toFixed(0)}%`
                  }
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="cost"
                >
                  {costByModel.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number) => `$${value.toFixed(2)}`} />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Token Usage Over Time */}
        <Card variant="elevated" className="lg:col-span-2">
          <CardHeader className="bg-gradient-to-r from-card to-sidebar/30">
            <CardTitle>Token Usage (24h)</CardTitle>
          </CardHeader>
          <CardContent className="pt-6">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={tokenUsage}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="input"
                  stroke="#3b82f6"
                  name="Input Tokens"
                  strokeWidth={2}
                  dot={{ fill: "#3b82f6", strokeWidth: 2 }}
                />
                <Line
                  type="monotone"
                  dataKey="output"
                  stroke="#22c55e"
                  name="Output Tokens"
                  strokeWidth={2}
                  dot={{ fill: "#22c55e", strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Budget Progress */}
      <Card variant="elevated">
        <CardHeader className="bg-gradient-to-r from-card to-sidebar/30">
          <CardTitle>Budget Usage</CardTitle>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>${totalCost.toFixed(2)} used</span>
              <span>${budgetLimit.toFixed(2)} limit</span>
            </div>
            <div className="h-4 bg-muted rounded-full overflow-hidden shadow-inner">
              <div
                className={`h-full transition-all duration-500 ${
                  budgetUsedPercent > 80
                    ? "bg-gradient-to-r from-red-500 to-red-600"
                    : budgetUsedPercent > 50
                    ? "bg-gradient-to-r from-yellow-500 to-yellow-600"
                    : "bg-gradient-to-r from-green-500 to-green-600"
                }`}
                style={{ width: `${Math.min(budgetUsedPercent, 100)}%` }}
              />
            </div>
            <p className="text-sm text-muted-foreground">
              {budgetUsedPercent > 80
                ? "Warning: Approaching budget limit!"
                : `${(100 - budgetUsedPercent).toFixed(1)}% remaining`}
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
