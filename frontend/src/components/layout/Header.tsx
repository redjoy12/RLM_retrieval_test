/** Header Component */

import { useState } from "react";
import { Settings, BarChart3, Bell, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SettingsPanel } from "@/components/settings/SettingsPanel";
import { CostDashboard } from "@/components/costs/CostDashboard";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

export function Header() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [costsOpen, setCostsOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 h-16 border-b bg-gradient-header shadow-elevation-sm backdrop-blur-sm flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        <h1 className="text-xl font-semibold bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent">
          RLM Document Retrieval
        </h1>
      </div>

      <div className="flex items-center gap-2">
        <Dialog open={costsOpen} onOpenChange={setCostsOpen}>
          <DialogTrigger asChild>
            <Button variant="ghost" size="icon" className="hover:bg-accent hover:shadow-elevation-sm transition-all">
              <BarChart3 className="h-5 w-5" />
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-4xl max-h-[80vh] overflow-auto">
            <DialogHeader>
              <DialogTitle>Cost Dashboard</DialogTitle>
            </DialogHeader>
            <CostDashboard />
          </DialogContent>
        </Dialog>

        <Button variant="ghost" size="icon" className="hover:bg-accent hover:shadow-elevation-sm transition-all">
          <Bell className="h-5 w-5" />
        </Button>

        <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
          <DialogTrigger asChild>
            <Button variant="ghost" size="icon" className="hover:bg-accent hover:shadow-elevation-sm transition-all">
              <Settings className="h-5 w-5" />
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl max-h-[80vh] overflow-auto">
            <DialogHeader>
              <DialogTitle>Settings</DialogTitle>
            </DialogHeader>
            <SettingsPanel />
          </DialogContent>
        </Dialog>

        <Button variant="ghost" size="icon" className="hover:bg-accent hover:shadow-elevation-sm transition-all">
          <User className="h-5 w-5" />
        </Button>
      </div>
    </header>
  );
}
