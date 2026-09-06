"use client";

import React, { useState } from "react";
import { ChatMessage as ChatMessageType, SourceReference } from "../../types/chat";
import { cn } from "../../lib/utils";
import { Avatar } from "../ui/Avatar";
import { CodeBlock } from "../ui/CodeBlock";
import {
  Copy,
  Check,
  FileText,
  ChevronDown,
  ChevronUp,
  Layers,
  Edit3,
  RotateCw,
  ThumbsUp,
  ThumbsDown,
  X,
} from "lucide-react";

interface ChatMessageProps {
  message: ChatMessageType;
  isStreaming?: boolean;
  isLastAssistant?: boolean;
  onEdit?: (messageId: string, newText: string) => void;
  onRegenerate?: () => void;
  onFeedback?: (messageId: string, feedback: "like" | "dislike") => void;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  isStreaming,
  isLastAssistant,
  onEdit,
  onRegenerate,
  onFeedback,
}) => {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const [showSources, setShowSources] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [editedText, setEditedText] = useState(message.content);

  // Normalize sources
  const sourcesList: SourceReference[] = Array.isArray(message.sources)
    ? message.sources
    : (message.sources as any)?.items || [];

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
    }
  };

  const handleSaveEdit = () => {
    if (!editedText.trim() || editedText.trim() === message.content) {
      setIsEditing(false);
      return;
    }
    if (onEdit) {
      onEdit(message.id, editedText.trim());
    }
    setIsEditing(false);
  };

  const handleCancelEdit = () => {
    setEditedText(message.content);
    setIsEditing(false);
  };

  // Helper to render markdown code blocks and paragraphs
  const renderFormattedContent = (content: string) => {
    if (!content) {
      return (
        <div className="flex items-center gap-1.5 py-1 text-emerald-400">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-bounce" />
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-bounce [animation-delay:0.2s]" />
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-bounce [animation-delay:0.4s]" />
        </div>
      );
    }

    // Split by code blocks ```lang ... ```
    const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(content)) !== null) {
      // Text before code block
      if (match.index > lastIndex) {
        parts.push({
          type: "text",
          content: content.substring(lastIndex, match.index),
        });
      }

      // Code block
      parts.push({
        type: "code",
        language: match[1] || "text",
        content: match[2].trim(),
      });

      lastIndex = match.index + match[0].length;
    }

    // Remaining text
    if (lastIndex < content.length) {
      parts.push({
        type: "text",
        content: content.substring(lastIndex),
      });
    }

    return parts.map((part, index) => {
      if (part.type === "code") {
        return (
          <CodeBlock
            key={index}
            language={part.language}
            code={part.content}
          />
        );
      }

      // Helper to strip # and * symbols and render clean text
      const renderCleanText = (text: string) => {
        const lines = text.split("\n");
        return lines.map((line, lineIdx) => {
          const isHeading = /^#{1,6}\s+/.test(line);
          let cleanLine = line.replace(/^#{1,6}\s+/, "");
          cleanLine = cleanLine.replace(/^(\s*)\*\s+/gm, "$1• ");
          const tokens = cleanLine.split(/(\*\*[^*]+?\*\*|\*[^*]+?\*)/g);

          return (
            <div
              key={lineIdx}
              className={cn(
                "min-h-[1.3rem] leading-relaxed",
                isHeading ? "font-semibold text-white text-base md:text-lg mt-2 mb-1" : ""
              )}
            >
              {tokens.map((token, tokenIdx) => {
                if (token.startsWith("**") && token.endsWith("**") && token.length >= 4) {
                  return (
                    <strong key={tokenIdx} className="font-semibold text-white">
                      {token.slice(2, -2).replace(/[\*#]/g, "")}
                    </strong>
                  );
                }
                if (token.startsWith("*") && token.endsWith("*") && token.length >= 2) {
                  return (
                    <span key={tokenIdx} className="font-medium text-gray-100">
                      {token.slice(1, -1).replace(/[\*#]/g, "")}
                    </span>
                  );
                }
                return <span key={tokenIdx}>{token.replace(/[\*#]/g, "")}</span>;
              })}
            </div>
          );
        });
      };

      return (
        <div
          key={index}
          className="space-y-1 text-sm md:text-base text-gray-200 break-words"
        >
          {renderCleanText(part.content)}
        </div>
      );
    });
  };

  return (
    <div
      className={cn(
        "group py-5 px-4 md:px-6 w-full flex justify-center border-b border-gray-800/30 transition-colors",
        isUser ? "bg-transparent" : "bg-[#212121]/40"
      )}
    >
      <div className="max-w-3xl w-full flex gap-4">
        {/* Role Avatar */}
        <Avatar role={message.role} size="md" />

        {/* Message Body */}
        <div className="flex-1 min-w-0 space-y-2.5">
          {/* Header info & Top Controls */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-xs text-gray-300">
                {isUser ? "You" : "Assistant"}
              </span>
              <span className="text-[10px] text-gray-500">
                {new Date(message.created_at * 1000 || Date.now()).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
            </div>

            {/* Action buttons on hover */}
            <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-opacity">
              {/* Copy button */}
              <button
                onClick={handleCopy}
                className="p-1.5 rounded-md text-gray-400 hover:text-gray-200 hover:bg-gray-800 text-xs flex items-center gap-1 transition-colors"
                title="Copy message"
              >
                {copied ? (
                  <>
                    <Check size={13} className="text-emerald-400" />
                    <span className="text-[10px] text-emerald-400">Copied</span>
                  </>
                ) : (
                  <Copy size={13} />
                )}
              </button>

              {/* Edit button (User message only) */}
              {isUser && !isEditing && (
                <button
                  onClick={() => setIsEditing(true)}
                  className="p-1.5 rounded-md text-gray-400 hover:text-gray-200 hover:bg-gray-800 text-xs transition-colors"
                  title="Edit prompt"
                >
                  <Edit3 size={13} />
                </button>
              )}

              {/* Regenerate button (Assistant message only) */}
              {!isUser && !isStreaming && onRegenerate && (
                <button
                  onClick={onRegenerate}
                  className="p-1.5 rounded-md text-gray-400 hover:text-emerald-400 hover:bg-gray-800 text-xs transition-colors"
                  title="Regenerate response"
                >
                  <RotateCw size={13} />
                </button>
              )}

              {/* Feedback controls (Assistant message only) */}
              {!isUser && !isStreaming && onFeedback && (
                <div className="flex items-center gap-0.5 border-l border-gray-700/60 pl-1 ml-0.5">
                  <button
                    onClick={() => onFeedback(message.id, "like")}
                    className={cn(
                      "p-1.5 rounded-md text-xs transition-colors",
                      message.feedback === "like"
                        ? "text-emerald-400 bg-emerald-950/60"
                        : "text-gray-400 hover:text-emerald-400 hover:bg-gray-800"
                    )}
                    title="Good response"
                  >
                    <ThumbsUp size={13} />
                  </button>
                  <button
                    onClick={() => onFeedback(message.id, "dislike")}
                    className={cn(
                      "p-1.5 rounded-md text-xs transition-colors",
                      message.feedback === "dislike"
                        ? "text-rose-400 bg-rose-950/60"
                        : "text-gray-400 hover:text-rose-400 hover:bg-gray-800"
                    )}
                    title="Poor response"
                  >
                    <ThumbsDown size={13} />
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Content (or Inline Edit Form) */}
          {isEditing ? (
            <div className="space-y-2 pt-1 animate-in fade-in">
              <textarea
                value={editedText}
                onChange={(e) => setEditedText(e.target.value)}
                className="w-full bg-[#181818] border border-emerald-500/80 rounded-xl p-3 text-xs md:text-sm text-gray-100 focus:outline-none resize-y min-h-[80px]"
                autoFocus
              />
              <div className="flex items-center justify-end gap-2">
                <button
                  onClick={handleCancelEdit}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold text-gray-400 hover:text-gray-200 hover:bg-gray-800 transition-colors flex items-center gap-1"
                >
                  <X size={12} />
                  <span>Cancel</span>
                </button>
                <button
                  onClick={handleSaveEdit}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition-colors flex items-center gap-1 shadow-sm"
                >
                  <Check size={12} />
                  <span>Save & Resend</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              {renderFormattedContent(message.content)}
              {isStreaming && message.content && (
                <span className="inline-block w-1.5 h-4 ml-1 bg-emerald-400 animate-pulse align-middle" />
              )}
            </div>
          )}

          {/* Grounded Sources / Citations */}
          {sourcesList.length > 0 && !isEditing && (
            <div className="mt-4 pt-3 border-t border-gray-800/80 space-y-2">
              <button
                onClick={() => setShowSources((prev) => !prev)}
                className="flex items-center gap-2 text-xs font-semibold text-gray-400 hover:text-emerald-400 transition-colors"
              >
                <Layers size={14} className="text-emerald-400" />
                <span>Grounded Citations ({sourcesList.length})</span>
                {showSources ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
              </button>

              {showSources && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1 animate-in fade-in">
                  {sourcesList.map((src, idx) => (
                    <div
                      key={idx}
                      className="bg-[#1a1a1a] rounded-xl p-3 text-xs border border-gray-800 hover:border-gray-700 transition-colors flex flex-col gap-1.5 shadow-sm"
                    >
                      <div className="flex items-center justify-between font-medium">
                        <span className="truncate text-gray-200 flex items-center gap-1.5">
                          <FileText size={12} className="text-emerald-400 shrink-0" />
                          <span className="truncate">{src.source}</span>
                        </span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-800/50 shrink-0 font-semibold">
                          {src.score_pct || `${Math.round(src.score * 100)}%`} match
                        </span>
                      </div>
                      <p className="text-gray-400 text-[11px] line-clamp-3 leading-relaxed">
                        {src.text}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

