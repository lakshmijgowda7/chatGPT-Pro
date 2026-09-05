"use client";

import React, { useState, useRef, useEffect } from "react";
import { ArrowUp, Paperclip, Square, Sparkles, Database, Lock } from "lucide-react";
import { cn } from "../../lib/utils";
import { useAuth } from "../../lib/AuthContext";
import { GUEST_MAX_CHATS } from "../../hooks/useChat";

interface ChatInputProps {
  onSend: (text: string, mode: "chat" | "rag") => void;
  onStop?: () => void;
  disabled?: boolean;
  isStreaming?: boolean;
  mode: "chat" | "rag";
  onToggleMode: () => void;
  onOpenUpload: () => void;
  stagedText?: string;
  onClearStagedText?: () => void;
  onOpenAuth?: () => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  onStop,
  disabled,
  isStreaming,
  mode,
  onToggleMode,
  onOpenUpload,
  stagedText,
  onClearStagedText,
  onOpenAuth,
}) => {
  const { user } = useAuth();
  const isGuest = Boolean(
    user?.profile?.is_anonymous ||
    user?.id?.startsWith("guest_") ||
    user?.email?.startsWith("guest_") ||
    user?.email?.toLowerCase().includes("guest")
  );

  const [guestCount, setGuestCount] = useState<number>(0);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = parseInt(localStorage.getItem("localgpt_guest_chat_count") || "0", 10);
      setGuestCount(saved);
    }
  }, [isStreaming]);

  const guestRemaining = Math.max(0, GUEST_MAX_CHATS - guestCount);
  const isGuestLimitReached = isGuest && guestRemaining <= 0;

  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Sync staged text if passed from suggestions
  useEffect(() => {
    if (stagedText) {
      setText(stagedText);
      if (onClearStagedText) onClearStagedText();
      textareaRef.current?.focus();
    }
  }, [stagedText, onClearStagedText]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [text]);

  const handleSubmit = () => {
    if (isGuestLimitReached) {
      if (onOpenAuth) onOpenAuth();
      return;
    }
    if (!text.trim() || isStreaming || disabled) return;
    onSend(text.trim(), mode);
    setText("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="p-4 bg-[#212121] border-t border-gray-800/80 max-w-4xl mx-auto w-full">
      {/* Guest Chat Remaining Counter Banner */}
      {isGuest && (
        <div className="mb-2.5 flex items-center justify-between px-3.5 py-1.5 rounded-xl bg-[#2a2a2a] border border-amber-500/30 text-xs">
          <div className="flex items-center gap-2 text-amber-300">
            <Sparkles size={14} className="text-amber-400" />
            <span>
              Guest Pass:{" "}
              <strong className="text-white">
                {guestRemaining} of {GUEST_MAX_CHATS}
              </strong>{" "}
              chats remaining
            </span>
          </div>
          {onOpenAuth && (
            <button
              onClick={onOpenAuth}
              className="text-[11px] font-bold px-2.5 py-0.5 rounded-md bg-amber-400/20 hover:bg-amber-400/30 text-amber-300 transition-colors"
            >
              Sign In for Unlimited
            </button>
          )}
        </div>
      )}

      {/* Main Input Box */}
      <div className="relative flex items-end gap-2 bg-[#2f2f2f] border border-gray-700/80 rounded-2xl p-2 shadow-lg focus-within:border-emerald-500/80 transition-colors">
        {/* Upload Knowledge Document Button */}
        <button
          type="button"
          onClick={onOpenUpload}
          className="p-2 text-gray-400 hover:text-emerald-400 rounded-xl hover:bg-[#383838] transition-colors"
          title="Upload knowledge document (PDF, TXT, DOCX)"
        >
          <Paperclip size={18} />
        </button>

        {/* Input Textarea */}
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            isGuestLimitReached
              ? "Guest chat limit reached (5/5). Click 'Sign In' above to unlock unlimited chats."
              : mode === "rag"
              ? "Ask questions grounded in indexed documents..."
              : "Message ChatGPT Pro..."
          }
          rows={1}
          className="flex-1 bg-transparent border-0 text-sm text-gray-100 placeholder-gray-500 focus:outline-none resize-none px-2 py-2 max-h-48 overflow-y-auto leading-relaxed"
          disabled={disabled || isStreaming || isGuestLimitReached}
        />

        {/* Action Button (Send / Stop / Limit Lock) */}
        {isStreaming ? (
          <button
            type="button"
            onClick={onStop}
            className="rounded-xl w-9 h-9 p-0 shrink-0 bg-gray-800 hover:bg-gray-700 text-gray-200 flex items-center justify-center transition-colors border border-gray-700"
            title="Stop generating"
          >
            <Square size={14} className="fill-current text-rose-400" />
          </button>
        ) : isGuestLimitReached ? (
          <button
            type="button"
            onClick={onOpenAuth}
            className="rounded-xl w-9 h-9 p-0 shrink-0 bg-amber-500 hover:bg-amber-400 text-black flex items-center justify-center transition-all shadow-md font-bold"
            title="Sign in to unlock unlimited chats"
          >
            <Lock size={15} />
          </button>
        ) : (
          <button
            type="button"
            onClick={() => handleSubmit()}
            disabled={!text.trim() || disabled}
            className={cn(
              "rounded-xl w-9 h-9 p-0 shrink-0 flex items-center justify-center transition-all shadow-md",
              text.trim() && !disabled
                ? "bg-gradient-to-r from-emerald-600 to-teal-600 text-white hover:from-emerald-500 hover:to-teal-500 active:scale-95 shadow-emerald-950/40"
                : "bg-gray-800 text-gray-600 cursor-not-allowed"
            )}
            title="Send message (Enter)"
          >
            <ArrowUp size={17} />
          </button>
        )}
      </div>

      {/* Footer Info */}
      <div className="text-[11px] text-center text-gray-500 mt-2 select-none">
        ChatGPT Pro may produce inaccurate information about people, places, or facts.
      </div>
    </div>
  );
};
