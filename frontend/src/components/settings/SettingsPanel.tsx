/** Settings Panel Component */

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const models = [
  { value: "gpt-5-mini", label: "GPT-5 Mini" },
  { value: "gpt-5", label: "GPT-5" },
  { value: "claude-sonnet", label: "Claude Sonnet 4.5" },
  { value: "claude-opus", label: "Claude Opus 4.5" },
] as const;

export function SettingsPanel() {
  const [settings, setSettings] = useState({
    // LLM Settings
    defaultModel: "gpt-5-mini",
    temperature: 0.7,
    maxTokens: 4000,
    
    // RLM Settings
    maxRecursionDepth: 3,
    timeoutPerBlock: 30,
    subLlmLimit: 100,
    
    // Cost Settings
    budgetLimit: 10.0,
    enableCostAlerts: true,
    
    // API Keys
    openaiKey: "",
    anthropicKey: "",
  });

  const handleSave = () => {
    // Save settings to backend
    console.log("Saving settings:", settings);
  };

  return (
    <Tabs defaultValue="llm" className="w-full">
      <TabsList className="grid w-full grid-cols-4 shadow-elevation-sm">
        <TabsTrigger value="llm">LLM</TabsTrigger>
        <TabsTrigger value="rlm">RLM</TabsTrigger>
        <TabsTrigger value="cost">Cost</TabsTrigger>
        <TabsTrigger value="api">API Keys</TabsTrigger>
      </TabsList>

      <TabsContent value="llm" className="space-y-4 mt-4">
        <Card variant="elevated">
          <CardHeader className="bg-gradient-to-r from-card to-sidebar/30">
            <CardTitle>LLM Configuration</CardTitle>
            <CardDescription>
              Configure default language model settings
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 pt-6">
            <div className="space-y-2">
              <label className="text-sm font-medium">Default Model</label>
              <Select
                value={settings.defaultModel}
                onValueChange={(value) =>
                  setSettings({ ...settings, defaultModel: value })
                }
              >
                <SelectTrigger className="w-full shadow-elevation-sm">
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {models.map((model) => (
                    <SelectItem key={model.value} value={model.value}>
                      {model.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Temperature</label>
              <Input
                type="number"
                min={0}
                max={2}
                step={0.1}
                value={settings.temperature}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    temperature: parseFloat(e.target.value),
                  })
                }
                className="shadow-elevation-sm"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Max Tokens</label>
              <Input
                type="number"
                min={100}
                max={8000}
                step={100}
                value={settings.maxTokens}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    maxTokens: parseInt(e.target.value),
                  })
                }
                className="shadow-elevation-sm"
              />
            </div>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="rlm" className="space-y-4 mt-4">
        <Card variant="elevated">
          <CardHeader className="bg-gradient-to-r from-card to-sidebar/30">
            <CardTitle>RLM Configuration</CardTitle>
            <CardDescription>
              Configure recursive language model behavior
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 pt-6">
            <div className="space-y-2">
              <label className="text-sm font-medium">Max Recursion Depth</label>
              <Input
                type="number"
                min={1}
                max={5}
                value={settings.maxRecursionDepth}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    maxRecursionDepth: parseInt(e.target.value),
                  })
                }
                className="shadow-elevation-sm"
              />
              <p className="text-xs text-muted-foreground">
                Maximum depth for recursive LLM calls
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">
                Timeout per Code Block (seconds)
              </label>
              <Input
                type="number"
                min={5}
                max={120}
                value={settings.timeoutPerBlock}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    timeoutPerBlock: parseInt(e.target.value),
                  })
                }
                className="shadow-elevation-sm"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Sub-LLM Call Limit</label>
              <Input
                type="number"
                min={10}
                max={500}
                value={settings.subLlmLimit}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    subLlmLimit: parseInt(e.target.value),
                  })
                }
                className="shadow-elevation-sm"
              />
            </div>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="cost" className="space-y-4 mt-4">
        <Card variant="elevated">
          <CardHeader className="bg-gradient-to-r from-card to-sidebar/30">
            <CardTitle>Cost Management</CardTitle>
            <CardDescription>
              Configure cost tracking and budget limits
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 pt-6">
            <div className="space-y-2">
              <label className="text-sm font-medium">Budget Limit (USD)</label>
              <Input
                type="number"
                min={0}
                step={0.01}
                value={settings.budgetLimit}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    budgetLimit: parseFloat(e.target.value),
                  })
                }
                className="shadow-elevation-sm"
              />
            </div>

            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="costAlerts"
                checked={settings.enableCostAlerts}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    enableCostAlerts: e.target.checked,
                  })
                }
                className="rounded border-gray-300 h-4 w-4"
              />
              <label htmlFor="costAlerts" className="text-sm font-medium">
                Enable cost alerts
              </label>
            </div>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="api" className="space-y-4 mt-4">
        <Card variant="elevated">
          <CardHeader className="bg-gradient-to-r from-card to-sidebar/30">
            <CardTitle>API Keys</CardTitle>
            <CardDescription>
              Configure API keys for LLM providers
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 pt-6">
            <div className="space-y-2">
              <label className="text-sm font-medium">OpenAI API Key</label>
              <Input
                type="password"
                placeholder="sk-..."
                value={settings.openaiKey}
                onChange={(e) =>
                  setSettings({ ...settings, openaiKey: e.target.value })
                }
                className="shadow-elevation-sm"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Anthropic API Key</label>
              <Input
                type="password"
                placeholder="sk-ant-..."
                value={settings.anthropicKey}
                onChange={(e) =>
                  setSettings({ ...settings, anthropicKey: e.target.value })
                }
                className="shadow-elevation-sm"
              />
            </div>
          </CardContent>
        </Card>
      </TabsContent>

      <div className="mt-6 flex justify-end">
        <Button 
          onClick={handleSave}
          className="shadow-elevation-sm hover:shadow-elevation-md transition-shadow"
        >
          Save Settings
        </Button>
      </div>
    </Tabs>
  );
}
