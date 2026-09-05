"use client";

import React, { useState } from "react";
import { Menu, Sparkles, Database, Plus, RotateCcw, ChevronDown, Check, Zap } from "lucide-react";
import { Badge } from "../ui/Badge";

interface ChatHeaderProps {
  onOpenMobileSidebar: () => void;
  mode: "chat" | "rag";
  onToggleMode: () => void;
  onNewChat: () => void;
  onClearMessages?: () => void;
  activeTitle?: string;
  hasMessages: boolean;
}

export const ChatHeader: React.FC<ChatHeaderProps> = ({
  onOpenMobileSidebar,
  mode,
  onToggleMode,
  onNewChat,
  onClearMessages,
  activeTitle,
  hasMessages,
}) => {
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<"gpt4o" | "fast">("gpt4o");

  return (
    <header className="h-14 border-b border-gray-800 bg-[#212121]/90 backdrop-blur-md px-4 flex items-center justify-between shrink-0 z-20 select-none">
      <div className="flex items-center gap-3 min-w-0">
        {/* Mobile menu trigger */}
        <button
          onClick={onOpenMobileSidebar}
          className="md:hidden p-2 -ml-1 text-gray-400 hover:text-white rounded-lg hover:bg-gray-800 focus:outline-none"
          aria-label="Open sidebar"
        >
          <Menu size={20} />
        </button>

        {/* ChatGPT Pro Model Selector */}
        <div className="relative">
          <button
            onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-xl hover:bg-white/5 transition-colors group"
          >
            <div className="flex items-center gap-1.5 font-bold text-sm text-gray-100">
              <Sparkles size={16} className="text-emerald-400 shrink-0" />
              <span>ChatGPT</span>
              <span className="px-1.5 py-0.5 rounded-md bg-gradient-to-r from-amber-400 to-orange-500 text-[10px] font-black text-black">
                PRO
              </span>
            </div>
            <ChevronDown
              size={14}
              className={`text-gray-400 transition-transform ${
                isModelDropdownOpen ? "rotate-180 text-white" : "group-hover:text-gray-200"
              }`}
            />
          </button>

          {/* Model Switcher Dropdown */}
          {isModelDropdownOpen && (
            <>
              <div
                className="fixed inset-0 z-30"
                onClick={() => setIsModelDropdownOpen(false)}
              />
              <div className="absolute top-11 left-0 z-40 w-64 rounded-2xl border border-gray-700/80 bg-[#1e1e1e] p-1.5 shadow-2xl shadow-black/60 backdrop-blur-xl animate-in fade-in zoom-in-95 duration-150">
                <div className="px-3 py-1.5 text-[10px] font-semibold text-gray-400 tracking-wider uppercase">
                  Pro Model Engine
                </div>

                <button
                  onClick={() => {
                    setSelectedModel("gpt4o");
                    setIsModelDropdownOpen(false);
                  }}
                  className={`w-full flex items-center justify-between p-2.5 rounded-xl text-left text-xs transition-colors ${
                    selectedModel === "gpt4o"
                      ? "bg-white/10 text-white font-semibold"
                      : "text-gray-300 hover:bg-white/5"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Sparkles size={15} className="text-emerald-400" />
                    <div>
                      <div className="font-semibold">GPT-4o Pro</div>
                      <div className="text-[10px] text-gray-400">Advanced reasoning & multimodal</div>
                    </div>
                  </div>
                  {selectedModel === "gpt4o" && <Check size={14} className="text-emerald-400" />}
                </button>

                <button
                  onClick={() => {
                    setSelectedModel("fast");
                    setIsModelDropdownOpen(false);
                  }}
                  className={`w-full flex items-center justify-between p-2.5 rounded-xl text-left text-xs transition-colors ${
                    selectedModel === "fast"
                      ? "bg-white/10 text-white font-semibold"
                      : "text-gray-300 hover:bg-white/5"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Zap size={15} className="text-amber-400" />
                    <div>
                      <div className="font-semibold">Qwen 2.5 Turbo</div>
                      <div className="text-[10px] text-gray-400">High-speed conversational stream</div>
                    </div>
                  </div>
                  {selectedModel === "fast" && <Check size={14} className="text-emerald-400" />}
                </button>
              </div>
            </>
          )}
        </div>

        {activeTitle && (
          <span className="hidden lg:inline text-xs text-gray-500 truncate max-w-xs border-l border-gray-700 pl-3">
            {activeTitle}
          </span>
        )}
      </div>

      {/* Right Header Controls */}
      <div className="flex items-center gap-2">
        {/* Mode Toggle Button */}
        <button
          onClick={onToggleMode}
          type="button"
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold transition-all border ${
            mode === "rag"
              ? "bg-emerald-950/70 text-emerald-300 border-emerald-700/60 shadow-sm shadow-emerald-950/50"
              : "bg-[#2a2a2a] text-gray-300 border-gray-700 hover:text-white hover:bg-[#333333]"
          }`}
          title="Toggle between General Chat and Document RAG mode"
        >
          {mode === "rag" ? (
            <>
              <Database size={13} className="text-emerald-400" />
              <span>Knowledge Search</span>
            </>
          ) : (
            <>
              <Sparkles size={13} className="text-gray-400" />
              <span>General Chat</span>
            </>
          )}
        </button>

        {/* Action: Clear current view */}
        {hasMessages && onClearMessages && (
          <button
            onClick={onClearMessages}
            className="p-1.5 text-gray-400 hover:text-gray-200 rounded-lg hover:bg-gray-800 transition-colors hidden sm:flex items-center gap-1 text-xs"
            title="Clear current messages"
          >
            <RotateCcw size={14} />
          </button>
        )}

        <button
          onClick={onNewChat}
          className="md:hidden p-2 text-gray-400 hover:text-emerald-400 rounded-lg hover:bg-gray-800"
          title="New Chat"
        >
          <Plus size={18} />
        </button>
      </div>
    </header>
  );
};
