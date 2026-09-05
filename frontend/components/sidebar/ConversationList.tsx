"use client";

import React, { useState } from "react";
import { ConversationSummary } from "../../types/chat";
import { MessageSquare, Edit2, Trash2, Check, X, Sparkles } from "lucide-react";
import { cn } from "../../lib/utils";

interface ConversationListProps {
  conversations: ConversationSummary[];
  activeId: string | null;
  searchQuery?: string;
  onSelect: (id: string) => void;
  onRename: (id: string, newTitle: string) => void;
  onDelete: (id: string) => void;
  onCreateNew?: () => void;
}

export const ConversationList: React.FC<ConversationListProps> = ({
  conversations,
  activeId,
  searchQuery = "",
  onSelect,
  onRename,
  onDelete,
  onCreateNew,
}) => {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const filteredConversations = conversations.filter((c) =>
    c.title.toLowerCase().includes(searchQuery.toLowerCase().trim())
  );

  const startEdit = (conv: ConversationSummary, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(conv.id);
    setEditTitle(conv.title);
    setDeletingId(null);
  };

  const saveEdit = (id: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    if (editTitle.trim()) {
      onRename(id, editTitle.trim());
    }
    setEditingId(null);
  };

  const cancelEdit = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(null);
  };

  const confirmDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    onDelete(id);
    setDeletingId(null);
  };

  const promptDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setDeletingId(id);
    setEditingId(null);
  };

  const cancelDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    setDeletingId(null);
  };

  if (filteredConversations.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-4 text-center text-gray-500 text-xs">
        {searchQuery ? (
          <div>
            <p>No conversations found matching</p>
            <p className="font-semibold text-gray-400 mt-0.5 truncate max-w-[180px]">
              "{searchQuery}"
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            <MessageSquare size={24} className="mx-auto text-gray-600 opacity-60" />
            <p>No chat history yet</p>
            {onCreateNew && (
              <button
                onClick={onCreateNew}
                className="text-xs text-emerald-400 hover:text-emerald-300 font-medium underline"
              >
                Start a new chat
              </button>
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-2 space-y-1 py-2">
      <div className="px-2 py-1 text-[11px] font-semibold tracking-wider text-gray-500 uppercase">
        {searchQuery ? "Search Results" : "Recent Chats"}
      </div>

      {filteredConversations.map((conv) => {
        const isActive = conv.id === activeId;
        const isEditing = conv.id === editingId;
        const isDeleting = conv.id === deletingId;

        return (
          <div
            key={conv.id}
            onClick={() => onSelect(conv.id)}
            className={cn(
              "group relative flex items-center justify-between px-3 py-2 rounded-xl text-xs transition-all cursor-pointer select-none",
              isActive
                ? "bg-[#252525] text-white font-medium shadow-sm border border-gray-700/50"
                : "text-gray-400 hover:bg-[#202020] hover:text-gray-200"
            )}
          >
            <div className="flex items-center gap-2.5 flex-1 min-w-0 mr-1.5">
              <MessageSquare
                size={15}
                className={cn(
                  "shrink-0 transition-colors",
                  isActive ? "text-emerald-400" : "text-gray-500 group-hover:text-gray-400"
                )}
              />

              {isEditing ? (
                <input
                  type="text"
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  onClick={(e) => e.stopPropagation()}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") saveEdit(conv.id);
                    if (e.key === "Escape") setEditingId(null);
                  }}
                  className="bg-[#2d2d2d] text-white px-2 py-0.5 rounded-lg text-xs w-full focus:outline-none border border-emerald-500/60"
                  autoFocus
                />
              ) : (
                <span className="truncate">{conv.title}</span>
              )}
            </div>

            {/* Action buttons */}
            <div className="flex items-center gap-1 shrink-0">
              {isEditing ? (
                <>
                  <button
                    onClick={(e) => saveEdit(conv.id, e)}
                    className="p-1 hover:text-emerald-400 text-gray-400 rounded hover:bg-gray-800"
                    title="Save"
                  >
                    <Check size={13} />
                  </button>
                  <button
                    onClick={cancelEdit}
                    className="p-1 hover:text-rose-400 text-gray-400 rounded hover:bg-gray-800"
                    title="Cancel"
                  >
                    <X size={13} />
                  </button>
                </>
              ) : isDeleting ? (
                <div className="flex items-center gap-1 bg-rose-950/80 border border-rose-800/80 rounded-lg px-1 py-0.5 animate-in fade-in">
                  <span className="text-[10px] text-rose-300 font-semibold px-1">Delete?</span>
                  <button
                    onClick={(e) => confirmDelete(conv.id, e)}
                    className="p-0.5 hover:text-white text-rose-300 rounded"
                    title="Confirm Delete"
                  >
                    <Check size={12} />
                  </button>
                  <button
                    onClick={cancelDelete}
                    className="p-0.5 hover:text-gray-200 text-gray-400 rounded"
                    title="Cancel"
                  >
                    <X size={12} />
                  </button>
                </div>
              ) : (
                <div className="opacity-0 group-hover:opacity-100 flex items-center gap-0.5 transition-opacity">
                  <button
                    onClick={(e) => startEdit(conv, e)}
                    className="p-1 hover:text-gray-100 text-gray-500 rounded hover:bg-gray-700/50"
                    title="Rename chat"
                  >
                    <Edit2 size={13} />
                  </button>
                  <button
                    onClick={(e) => promptDelete(conv.id, e)}
                    className="p-1 hover:text-rose-400 text-gray-500 rounded hover:bg-gray-700/50"
                    title="Delete chat"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};
