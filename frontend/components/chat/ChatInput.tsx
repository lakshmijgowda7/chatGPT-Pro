"use client";

import React, { useState, useRef, useEffect } from "react";
import { ArrowUp, Paperclip, Square } from "lucide-react";
import { cn } from "../../lib/utils";

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
  onOpenUpload,
  stagedText,
  onClearStagedText,
}) => {
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
            mode === "rag"
              ? "Ask questions grounded in indexed documents..."
              : "Message ChatGPT Pro..."
          }
          rows={1}
          className="flex-1 bg-transparent border-0 text-sm text-gray-100 placeholder-gray-500 focus:outline-none resize-none px-2 py-2 max-h-48 overflow-y-auto leading-relaxed"
          disabled={disabled || isStreaming}
        />

        {/* Action Button (Send / Stop) */}
        {isStreaming ? (
          <button
            type="button"
            onClick={onStop}
            className="rounded-xl w-9 h-9 p-0 shrink-0 bg-gray-800 hover:bg-gray-700 text-gray-200 flex items-center justify-center transition-colors border border-gray-700"
            title="Stop generating"
          >
            <Square size={14} className="fill-current text-rose-400" />
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
