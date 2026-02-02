/** Query Interface Component */

import { useState } from "react";
import { Send, Loader2, Bot, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
}

const strategies = [
  { value: "rlm", label: "RLM (Recursive)" },
  { value: "rag", label: "RAG (Retrieval)" },
  { value: "hybrid", label: "Hybrid" },
] as const;

export function QueryInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [strategy, setStrategy] = useState<"rlm" | "rag" | "hybrid">("rlm");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Simulate streaming response
    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: "assistant",
      content: "",
      timestamp: new Date(),
      isStreaming: true,
    };

    setMessages((prev) => [...prev, assistantMessage]);

    // Simulate streaming
    const response =
      "This is a simulated response from the RLM system. In production, this would be a real-time streaming response from the backend API using WebSocket connections.";
    let currentContent = "";

    for (let i = 0; i < response.length; i++) {
      await new Promise((resolve) => setTimeout(resolve, 30));
      currentContent += response[i];
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMessage.id
            ? { ...m, content: currentContent }
            : m
        )
      );
    }

    setMessages((prev) =>
      prev.map((m) =>
        m.id === assistantMessage.id ? { ...m, isStreaming: false } : m
      )
    );

    setIsLoading(false);
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <Card variant="elevated" className="w-full h-[calc(100vh-12rem)] flex flex-col">
      <CardHeader className="border-b flex flex-row items-center justify-between bg-gradient-to-r from-card to-sidebar/50">
        <CardTitle className="text-xl">Query Interface</CardTitle>
        <div className="flex items-center gap-2">
          <Select
            value={strategy}
            onValueChange={(value) => setStrategy(value as "rlm" | "rag" | "hybrid")}
          >
            <SelectTrigger className="w-[180px] h-9 shadow-elevation-sm">
              <SelectValue placeholder="Select strategy" />
            </SelectTrigger>
            <SelectContent>
              {strategies.map((s) => (
                <SelectItem key={s.value} value={s.value}>
                  {s.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={clearChat}
            className="shadow-elevation-sm hover:shadow-elevation-md transition-shadow"
          >
            Clear
          </Button>
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col p-0">
        <ScrollArea className="flex-1 p-4">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center space-y-4 animate-fade-in">
                <div className="w-20 h-20 mx-auto rounded-full bg-gradient-to-br from-primary/20 to-primary/10 flex items-center justify-center shadow-elevation-md">
                  <Bot className="h-10 w-10 text-primary" />
                </div>
                <div>
                  <p className="text-lg font-medium text-foreground">Start a conversation</p>
                  <p className="text-sm text-muted-foreground mt-2 max-w-sm">
                    Ask questions about your uploaded documents. The RLM system will provide intelligent, context-aware responses.
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div
                  key={message.id}
                  className={cn(
                    "flex gap-3 animate-slide-in",
                    message.role === "user" ? "justify-end" : "justify-start"
                  )}
                  style={{ animationDelay: `${index * 50}ms` }}
                >
                  {message.role === "assistant" && (
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary/20 to-primary/10 flex items-center justify-center flex-shrink-0 shadow-elevation-sm">
                      <Bot className="h-4 w-4 text-primary" />
                    </div>
                  )}
                  <div
                    className={cn(
                      "max-w-[80%] rounded-2xl px-4 py-3 shadow-elevation-sm",
                      message.role === "user"
                        ? "bg-gradient-to-br from-primary to-primary/90 text-primary-foreground"
                        : "bg-card border border-border/50"
                    )}
                  >
                    <p className="text-sm whitespace-pre-wrap leading-relaxed">
                      {message.content}
                      {message.isStreaming && (
                        <span className="inline-block w-2 h-4 bg-current ml-1 animate-pulse" />
                      )}
                    </p>
                    <p className="text-xs opacity-70 mt-2">
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                  {message.role === "user" && (
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary to-primary/80 flex items-center justify-center flex-shrink-0 shadow-elevation-sm">
                      <User className="h-4 w-4 text-primary-foreground" />
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </ScrollArea>

        <div className="border-t p-4 bg-gradient-to-r from-card to-sidebar/30">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about your documents..."
              disabled={isLoading}
              className="flex-1 shadow-elevation-sm"
            />
            <Button 
              type="submit" 
              disabled={isLoading || !input.trim()}
              className="shadow-elevation-sm hover:shadow-elevation-md transition-shadow"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </form>
        </div>
      </CardContent>
    </Card>
  );
}
