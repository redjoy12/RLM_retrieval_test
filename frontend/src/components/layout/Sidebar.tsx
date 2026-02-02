/** Sidebar Component */

import { useState } from "react";
import { FileText, MessageSquare, GitBranch, Database, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface NavItem {
  icon: React.ReactNode;
  label: string;
  value: string;
}

const navItems: NavItem[] = [
  { icon: <MessageSquare className="h-5 w-5" />, label: "Query", value: "query" },
  { icon: <FileText className="h-5 w-5" />, label: "Documents", value: "documents" },
  { icon: <GitBranch className="h-5 w-5" />, label: "Trajectories", value: "trajectory" },
  { icon: <Database className="h-5 w-5" />, label: "Sessions", value: "sessions" },
];

interface SidebarProps {
  activeTab?: string;
  onTabChange?: (tab: string) => void;
}

export function Sidebar({ activeTab = "query", onTabChange }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "border-r bg-sidebar flex flex-col transition-all duration-300 shadow-elevation-sm",
        collapsed ? "w-16" : "w-64"
      )}
    >
      <div className="h-16 border-b border-sidebar-border flex items-center justify-between px-4 bg-gradient-to-r from-sidebar to-sidebar-accent">
        {!collapsed && (
          <span className="font-semibold text-lg bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent">
            RLM
          </span>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setCollapsed(!collapsed)}
          className="ml-auto hover:bg-sidebar-accent hover:shadow-elevation-sm transition-all"
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      <nav className="flex-1 p-2 space-y-1">
        {navItems.map((item) => (
          <button
            key={item.value}
            onClick={() => onTabChange?.(item.value)}
            className={cn(
              "w-full flex items-center gap-3 px-3 py-2 rounded-md transition-all duration-200 text-left",
              activeTab === item.value
                ? "bg-primary text-primary-foreground shadow-elevation-sm"
                : "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground hover:shadow-elevation-sm"
            )}
          >
            {item.icon}
            {!collapsed && <span className="font-medium">{item.label}</span>}
          </button>
        ))}
      </nav>

      <div className="p-4 border-t border-sidebar-border">
        {!collapsed && (
          <div className="text-xs text-sidebar-foreground/60">
            <p className="font-medium">v0.1.0</p>
            <p className="mt-1">RLM Document Retrieval</p>
          </div>
        )}
      </div>
    </aside>
  );
}
