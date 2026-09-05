import React from "react";
import { Search, X } from "lucide-react";

interface SearchConversationsProps {
  query: string;
  onQueryChange: (q: string) => void;
  resultCount?: number;
}

export const SearchConversations: React.FC<SearchConversationsProps> = ({
  query,
  onQueryChange,
  resultCount,
}) => {
  return (
    <div className="px-3 py-1.5">
      <div className="relative flex items-center">
        <Search
          size={14}
          className="absolute left-2.5 text-gray-400 pointer-events-none"
        />
        <input
          type="text"
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          placeholder="Search conversations..."
          className="w-full bg-[#202020] hover:bg-[#252525] focus:bg-[#202020] border border-gray-700/60 focus:border-emerald-500/80 rounded-xl pl-8 pr-7 py-1.5 text-xs text-gray-200 placeholder-gray-500 focus:outline-none transition-all"
        />
        {query && (
          <button
            onClick={() => onQueryChange("")}
            className="absolute right-2 text-gray-400 hover:text-gray-200 p-0.5 rounded focus:outline-none"
            title="Clear search"
          >
            <X size={12} />
          </button>
        )}
      </div>
      {query.trim() && (
        <div className="mt-1.5 px-1 text-[11px] text-gray-400 flex justify-between items-center">
          <span>Matching chats:</span>
          <span className="font-semibold text-emerald-400">{resultCount ?? 0}</span>
        </div>
      )}
    </div>
  );
};
