/** Main App Component */

import { useState } from "react";
import { AppLayout } from "@/components/layout/AppLayout";
import { QueryInterface } from "@/components/query/QueryInterface";
import { DocumentUpload } from "@/components/upload/DocumentUpload";
import { TrajectoryViewer } from "@/components/trajectory/TrajectoryViewer";
import { SessionsList } from "@/components/sessions/SessionsList";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MessageSquare, FileText, GitBranch, Database } from "lucide-react";

function App() {
  const [activeTab, setActiveTab] = useState("query");

  return (
    <AppLayout activeTab={activeTab} onTabChange={setActiveTab}>
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4 lg:max-w-md">
          <TabsTrigger value="query" className="flex items-center gap-2">
            <MessageSquare className="h-4 w-4" />
            Query
          </TabsTrigger>
          <TabsTrigger value="documents" className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Documents
          </TabsTrigger>
          <TabsTrigger value="trajectory" className="flex items-center gap-2">
            <GitBranch className="h-4 w-4" />
            Trajectory
          </TabsTrigger>
          <TabsTrigger value="sessions" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            Sessions
          </TabsTrigger>
        </TabsList>

        <TabsContent value="query" className="mt-6">
          <QueryInterface />
        </TabsContent>

        <TabsContent value="documents" className="mt-6">
          <DocumentUpload />
        </TabsContent>

        <TabsContent value="trajectory" className="mt-6">
          <TrajectoryViewer />
        </TabsContent>

        <TabsContent value="sessions" className="mt-6">
          <SessionsList />
        </TabsContent>
      </Tabs>
    </AppLayout>
  );
}

export default App;
