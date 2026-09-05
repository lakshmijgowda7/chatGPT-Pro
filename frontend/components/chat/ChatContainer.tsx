"use client";

import React, { useState, useRef, useEffect } from "react";
import { ChatMessage as ChatMessageType } from "../../types/chat";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { ChatHeader } from "./ChatHeader";
import { EmptyState } from "./EmptyState";
import { DocumentUploader } from "../rag/DocumentUploader";
import { AuthModal } from "../auth/AuthModal";
import { ArrowDown } from "lucide-react";

interface ChatContainerProps {
  messages: ChatMessageType[];
  isStreaming: boolean;
  activeTitle?: string;
  onSend: (text: string, mode: "chat" | "rag") => void;
  onStop?: () => void;
  onRegenerate?: (mode?: "chat" | "rag") => void;
  onEdit?: (messageId: string, newText: string, mode?: "chat" | "rag") => void;
  onFeedback?: (messageId: string, feedback: "like" | "dislike") => void;
  onNewChat: () => void;
  onClearMessages?: () => void;
  onOpenMobileSidebar: () => void;
}

export const ChatContainer: React.FC<ChatContainerProps> = ({
  messages,
  isStreaming,
  activeTitle,
  onSend,
  onStop,
  onRegenerate,
  onEdit,
  onFeedback,
  onNewChat,
  onClearMessages,
  onOpenMobileSidebar,
}) => {

  const [mode, setMode] = useState<"chat" | "rag">("chat");
  const [isUploaderOpen, setIsUploaderOpen] = useState(false);
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [stagedText, setStagedText] = useState("");
  const [showScrollBottom, setShowScrollBottom] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Smooth scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  // Track scroll position to show/hide "Scroll to bottom" button
  const handleScroll = () => {
    if (!scrollContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 120;
    setShowScrollBottom(!isNearBottom && messages.length > 0);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSelectSuggestion = (promptText: string) => {
    setStagedText(promptText);
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#1e1e1e] overflow-hidden relative">
      {/* Top Header Bar */}
      <ChatHeader
        onOpenMobileSidebar={onOpenMobileSidebar}
        mode={mode}
        onToggleMode={() => setMode((prev) => (prev === "chat" ? "rag" : "chat"))}
        onNewChat={onNewChat}
        onClearMessages={onClearMessages}
        activeTitle={activeTitle}
        hasMessages={messages.length > 0}
      />

      {/* Main Messages & Empty State View */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto relative flex flex-col"
      >
        {messages.length === 0 ? (
          <EmptyState
            onSelectSuggestion={handleSelectSuggestion}
            mode={mode}
            onOpenUploader={() => setIsUploaderOpen(true)}
          />
        ) : (
          <div className="flex-1 pb-6">
            {messages.map((msg, index) => (
              <ChatMessage
                key={msg.id || index}
                message={msg}
                isStreaming={isStreaming && index === messages.length - 1 && msg.role === "assistant"}
                isLastAssistant={msg.role === "assistant" && index === messages.length - 1}
                onEdit={onEdit ? (id, text) => onEdit(id, text, mode) : undefined}
                onRegenerate={onRegenerate ? () => onRegenerate(mode) : undefined}
                onFeedback={onFeedback}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}

        {/* Scroll to Bottom Button */}
        {showScrollBottom && (
          <button
            onClick={scrollToBottom}
            className="fixed bottom-24 right-8 p-2.5 rounded-full bg-[#2a2a2a] hover:bg-[#333333] text-gray-200 shadow-xl border border-gray-700 transition-all animate-in fade-in zoom-in"
            title="Scroll to bottom"
          >
            <ArrowDown size={16} />
          </button>
        )}
      </div>

      {/* Message Input Toolbar */}
      <ChatInput
        onSend={onSend}
        onStop={onStop}
        disabled={isStreaming}
        isStreaming={isStreaming}
        mode={mode}
        onToggleMode={() => setMode((prev) => (prev === "chat" ? "rag" : "chat"))}
        onOpenUpload={() => setIsUploaderOpen(true)}
        stagedText={stagedText}
        onClearStagedText={() => setStagedText("")}
        onOpenAuth={() => setIsAuthOpen(true)}
      />

      {/* Document Uploader Modal */}
      <DocumentUploader
        isOpen={isUploaderOpen}
        onClose={() => setIsUploaderOpen(false)}
        onUploadSuccess={() => {}}
      />

      {/* Auth Modal for Upgrading from Guest Pass */}
      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
      />
    </div>
  );
};
