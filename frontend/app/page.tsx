"use client";

import React, { useState, useEffect } from "react";
import { Sidebar } from "../components/sidebar/Sidebar";
import { ChatContainer } from "../components/chat/ChatContainer";
import { AuthGate } from "../components/auth/AuthGate";
import { useConversations } from "../hooks/useConversations";
import { useChat } from "../hooks/useChat";
import { fetchSettings } from "../lib/api";
import { PlatformSettings } from "../types/api";
import { useAuth } from "../lib/AuthContext";
import { Sparkles } from "lucide-react";

export default function Home() {
  const { user, isAuthenticated, isLoading: isAuthLoading } = useAuth();
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [settings, setSettings] = useState<PlatformSettings | null>(null);

  const {
    conversations,
    createChat,
    renameChat,
    deleteChat,
    refresh: refreshConversations,
  } = useConversations(activeConversationId, setActiveConversationId);

  const {
    messages,
    isStreaming,
    sendMessage,
    stopGeneration,
    regenerate,
    editAndResend,
    setFeedback,
    clearMessages,
  } = useChat(
    activeConversationId,
    (newConvId) => {
      setActiveConversationId(newConvId);
      refreshConversations();
    },
    () => {
      refreshConversations();
    }
  );

  useEffect(() => {
    fetchSettings()
      .then(setSettings)
      .catch(() => {
        // Default settings state if backend is offline
        setSettings({
          project_name: "ChatGPT Pro Platform",
          llm_provider: "Cloud Inference (Groq / OpenAI)",
          llm_model: "gpt-4o-pro / qwen-2.5-72b",
          llm_base_url: "https://api.groq.com/openai/v1",
          masked_api_key: "sk-••••••••••••••••",
          default_temperature: 0.7,
          default_top_p: 0.95,
          default_max_tokens: 4096,
          debug_mode: false,
        });
      });
  }, []);

  // 1. If checking authentication state, display sleek loading splash
  if (isAuthLoading) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-[#181818] text-gray-100 font-sans">
        <div className="flex flex-col items-center gap-3 animate-in fade-in duration-300">
          <div className="relative flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-tr from-emerald-500 via-teal-400 to-cyan-500 p-0.5 shadow-2xl shadow-emerald-500/20">
            <div className="flex h-full w-full items-center justify-center rounded-[14px] bg-[#1e1e1e]">
              <Sparkles className="h-7 w-7 text-emerald-400 animate-pulse" />
            </div>
          </div>
          <span className="text-xs font-medium text-gray-400 tracking-wider">
            Loading ChatGPT Pro...
          </span>
        </div>
      </div>
    );
  }

  // 2. ENFORCED AUTH GATE: If user is not authenticated, show ChatGPT Pro login options first
  if (!isAuthenticated || !user) {
    return <AuthGate />;
  }

  const handleNewChat = () => {
    clearMessages();
    createChat();
  };

  const activeConversation = conversations.find((c) => c.id === activeConversationId);

  // 3. User is authenticated -> Open ChatGPT Pro Chat Box and Workspace
  return (
    <main className="flex h-screen w-screen bg-[#212121] text-gray-100 overflow-hidden font-sans">
      {/* Responsive Sidebar */}
      <Sidebar
        conversations={conversations}
        activeId={activeConversationId}
        isOpenMobile={isMobileSidebarOpen}
        onCloseMobile={() => setIsMobileSidebarOpen(false)}
        onSelect={setActiveConversationId}
        onCreate={handleNewChat}
        onRename={renameChat}
        onDelete={deleteChat}
        settings={settings}
      />

      {/* Main ChatGPT Pro Chat Interface */}
      <ChatContainer
        messages={messages}
        isStreaming={isStreaming}
        activeTitle={activeConversation?.title}
        onSend={sendMessage}
        onStop={stopGeneration}
        onRegenerate={regenerate}
        onEdit={editAndResend}
        onFeedback={setFeedback}
        onNewChat={handleNewChat}
        onClearMessages={clearMessages}
        onOpenMobileSidebar={() => setIsMobileSidebarOpen(true)}
      />
    </main>
  );
}
