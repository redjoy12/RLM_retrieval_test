/** Utility functions for the RLM Trajectory Visualizer */

import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import {
  TrajectoryStepType,
  TokenCost,
  TrajectoryNode,
} from "@/types/trajectory";

/** Merge Tailwind classes */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Format duration in milliseconds to human readable string */
export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "N/A";
  
  if (ms < 1000) {
    return `${ms.toFixed(0)}ms`;
  } else if (ms < 60000) {
    return `${(ms / 1000).toFixed(1)}s`;
  } else {
    const minutes = Math.floor(ms / 60000);
    const seconds = ((ms % 60000) / 1000).toFixed(1);
    return `${minutes}m ${seconds}s`;
  }
}

/** Format token count with commas */
export function formatTokens(tokens: number): string {
  return tokens.toLocaleString();
}

/** Format cost in USD */
export function formatCost(costUsd: number): string {
  if (costUsd < 0.01) {
    return `$${(costUsd * 100).toFixed(2)}¢`;
  }
  return `$${costUsd.toFixed(4)}`;
}

/** Format timestamp to local time */
export function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleString();
}

/** Get color for step type */
export function getStepTypeColor(type: TrajectoryStepType): string {
  const colors: Record<TrajectoryStepType, string> = {
    ROOT_LLM_START: "#3b82f6", // blue-500
    ROOT_LLM_COMPLETE: "#2563eb", // blue-600
    CODE_EXECUTION_START: "#eab308", // yellow-500
    CODE_EXECUTION_COMPLETE: "#ca8a04", // yellow-600
    SUB_LLM_SPAWN: "#22c55e", // green-500
    SUB_LLM_COMPLETE: "#16a34a", // green-600
    RECURSION_LIMIT_HIT: "#f97316", // orange-500
    ERROR: "#ef4444", // red-500
    FINAL_ANSWER: "#6b7280", // gray-500
  };
  return colors[type] || "#9ca3af";
}

/** Get background color class for step type */
export function getStepTypeBgClass(type: TrajectoryStepType): string {
  const classes: Record<TrajectoryStepType, string> = {
    ROOT_LLM_START: "bg-blue-500",
    ROOT_LLM_COMPLETE: "bg-blue-600",
    CODE_EXECUTION_START: "bg-yellow-500",
    CODE_EXECUTION_COMPLETE: "bg-yellow-600",
    SUB_LLM_SPAWN: "bg-green-500",
    SUB_LLM_COMPLETE: "bg-green-600",
    RECURSION_LIMIT_HIT: "bg-orange-500",
    ERROR: "bg-red-500",
    FINAL_ANSWER: "bg-gray-500",
  };
  return classes[type] || "bg-gray-400";
}

/** Get text color class for step type */
export function getStepTypeTextClass(type: TrajectoryStepType): string {
  const classes: Record<TrajectoryStepType, string> = {
    ROOT_LLM_START: "text-blue-500",
    ROOT_LLM_COMPLETE: "text-blue-600",
    CODE_EXECUTION_START: "text-yellow-600",
    CODE_EXECUTION_COMPLETE: "text-yellow-700",
    SUB_LLM_SPAWN: "text-green-500",
    SUB_LLM_COMPLETE: "text-green-600",
    RECURSION_LIMIT_HIT: "text-orange-500",
    ERROR: "text-red-500",
    FINAL_ANSWER: "text-gray-500",
  };
  return classes[type] || "text-gray-400";
}

/** Get human readable label for step type */
export function getStepTypeLabel(type: TrajectoryStepType): string {
  const labels: Record<TrajectoryStepType, string> = {
    ROOT_LLM_START: "Root LLM Start",
    ROOT_LLM_COMPLETE: "Root LLM Complete",
    CODE_EXECUTION_START: "Code Execution Start",
    CODE_EXECUTION_COMPLETE: "Code Execution Complete",
    SUB_LLM_SPAWN: "Sub-LLM Spawn",
    SUB_LLM_COMPLETE: "Sub-LLM Complete",
    RECURSION_LIMIT_HIT: "Recursion Limit Hit",
    ERROR: "Error",
    FINAL_ANSWER: "Final Answer",
  };
  return labels[type] || type;
}

/** Get icon name for step type */
export function getStepTypeIcon(type: TrajectoryStepType): string {
  const icons: Record<TrajectoryStepType, string> = {
    ROOT_LLM_START: "Brain",
    ROOT_LLM_COMPLETE: "Brain",
    CODE_EXECUTION_START: "Code",
    CODE_EXECUTION_COMPLETE: "Code",
    SUB_LLM_SPAWN: "GitBranch",
    SUB_LLM_COMPLETE: "GitBranch",
    RECURSION_LIMIT_HIT: "AlertTriangle",
    ERROR: "XCircle",
    FINAL_ANSWER: "CheckCircle",
  };
  return icons[type] || "Circle";
}

/** Calculate total cost from array of costs */
export function calculateTotalCost(costs: TokenCost[]): number {
  return costs.reduce((sum, cost) => sum + cost.cost_usd, 0);
}

/** Calculate total tokens from array of costs */
export function calculateTotalTokens(costs: TokenCost[]): number {
  return costs.reduce((sum, cost) => sum + cost.total_tokens, 0);
}

/** Truncate text with ellipsis */
export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + "...";
}

/** Build node label for React Flow */
export function buildNodeLabel(node: TrajectoryNode): string {
  const lines = [getStepTypeLabel(node.type)];
  
  if (node.duration_ms !== null && node.duration_ms !== undefined) {
    lines.push(`⏱️ ${formatDuration(node.duration_ms)}`);
  }
  
  if (node.cost.total_tokens > 0) {
    lines.push(`🪙 ${formatTokens(node.cost.total_tokens)} tokens`);
    lines.push(`💰 ${formatCost(node.cost.cost_usd)}`);
  }
  
  return lines.join("\n");
}

/** Download data as file */
export function downloadFile(
  content: string | Blob,
  filename: string,
  contentType: string = "text/plain"
): void {
  const blob =
    content instanceof Blob
      ? content
      : new Blob([content], { type: contentType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/** Debounce function */
export function debounce<T extends (...args: any[]) => void>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

/** Generate unique ID */
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/** Parse JSON safely */
export function safeJsonParse<T>(json: string, defaultValue: T): T {
  try {
    return JSON.parse(json) as T;
  } catch {
    return defaultValue;
  }
}
