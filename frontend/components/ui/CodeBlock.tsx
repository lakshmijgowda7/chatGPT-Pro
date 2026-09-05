"use client";

import React, { useState } from "react";
import { Check, Copy } from "lucide-react";

interface CodeBlockProps {
  code: string;
  language?: string;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({ code, language = "code" }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
    }
  };

  return (
    <div className="my-3 rounded-xl overflow-hidden border border-gray-800 bg-[#121212] shadow-md">
      <div className="flex items-center justify-between px-4 py-1.5 bg-[#1a1a1a] border-b border-gray-800/80 text-xs text-gray-400">
        <span className="font-mono lowercase text-gray-300 font-medium">{language}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-xs hover:text-gray-200 transition-colors py-0.5 px-2 rounded hover:bg-gray-800"
          title="Copy code"
        >
          {copied ? (
            <>
              <Check size={13} className="text-emerald-400" />
              <span className="text-emerald-400 text-[11px]">Copied!</span>
            </>
          ) : (
            <>
              <Copy size={13} />
              <span className="text-[11px]">Copy code</span>
            </>
          )}
        </button>
      </div>
      <pre className="p-4 overflow-x-auto text-xs font-mono text-gray-200 leading-relaxed">
        <code>{code}</code>
      </pre>
    </div>
  );
};
