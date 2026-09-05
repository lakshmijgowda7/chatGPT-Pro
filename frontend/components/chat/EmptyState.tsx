"use client";

import React from "react";
import { Sparkles, Code2, Compass, Brain, FileText, ArrowUpRight } from "lucide-react";

interface EmptyStateProps {
  onSelectSuggestion: (text: string) => void;
  mode: "chat" | "rag";
  onOpenUploader?: () => void;
}

const PRO_SUGGESTIONS = [
  {
    icon: <Code2 className="text-emerald-400 shrink-0" size={18} />,
    title: "Code Architecture",
    description: "Write a high-throughput async SSE streaming server in Python FastAPI.",
    prompt: "Write a high-throughput async SSE streaming server in Python FastAPI with backpressure handling.",
  },
  {
    icon: <Brain className="text-teal-400 shrink-0" size={18} />,
    title: "Deep Reasoning",
    description: "Explain how self-attention heads compute query-key dot products.",
    prompt: "Explain how multi-head self-attention computes query-key dot products with intuitive visual matrices.",
  },
  {
    icon: <FileText className="text-cyan-400 shrink-0" size={18} />,
    title: "Document Synthesis",
    description: "Compare vector embeddings vs sparse BM25 indexing in retrieval.",
    prompt: "Analyze the mathematical and computational trade-offs between dense vector embeddings and sparse BM25 indexing.",
  },
  {
    icon: <Compass className="text-amber-400 shrink-0" size={18} />,
    title: "Brainstorm Ideas",
    description: "Design a state-of-the-art AI application with Firebase & Cloud Run.",
    prompt: "Design a production-ready system architecture for a scalable multi-tenant AI platform using Firebase and Cloud Run.",
  },
];

export const EmptyState: React.FC<EmptyStateProps> = ({
  onSelectSuggestion,
}) => {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 md:p-10 text-center max-w-2xl mx-auto space-y-8 animate-in fade-in duration-300 my-auto select-none">
      {/* ChatGPT Pro Emblem */}
      <div className="flex flex-col items-center gap-3">
        <div className="relative flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-tr from-emerald-500 via-teal-400 to-cyan-500 p-0.5 shadow-2xl shadow-emerald-500/20">
          <div className="flex h-full w-full items-center justify-center rounded-[14px] bg-[#1e1e1e]">
            <Sparkles className="h-7 w-7 text-emerald-400" />
          </div>
          <span className="absolute -top-1 -right-2 px-1.5 py-0.5 rounded-full bg-gradient-to-r from-amber-400 to-orange-500 text-[9px] font-black text-black">
            PRO
          </span>
        </div>

        <h1 className="text-2xl md:text-3xl font-bold text-white tracking-tight">
          What can I help with today?
        </h1>
        <p className="text-xs md:text-sm text-gray-400 max-w-md">
          ChatGPT Pro offers high-speed token generation, vector-grounded document search, and deep reasoning.
        </p>
      </div>

      {/* Prompt Suggestion Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full pt-2">
        {PRO_SUGGESTIONS.map((item, idx) => (
          <button
            key={idx}
            onClick={() => onSelectSuggestion(item.prompt)}
            className="group flex flex-col justify-between p-3.5 text-left rounded-xl bg-[#2a2a2a]/60 hover:bg-[#323232] border border-gray-800 hover:border-gray-700 transition-all cursor-pointer shadow-sm hover:shadow-emerald-950/20"
          >
            <div className="flex items-center justify-between w-full mb-1.5">
              <div className="flex items-center gap-2">
                {item.icon}
                <span className="text-xs font-semibold text-gray-200 group-hover:text-emerald-300 transition-colors">
                  {item.title}
                </span>
              </div>
              <ArrowUpRight size={14} className="text-gray-500 group-hover:text-gray-300 transition-colors" />
            </div>
            <p className="text-[11px] text-gray-400 leading-relaxed line-clamp-2">
              {item.description}
            </p>
          </button>
        ))}
      </div>
    </div>
  );
};
