"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { ConversationSummary } from "../types/chat";
import {
  fetchConversations,
  createConversation,
  renameConversation,
  deleteConversation,
} from "../lib/api";

const DEFAULT_CONVERSATIONS: ConversationSummary[] = [
  {
    id: "conv-welcome",
    title: "Welcome to LocalGPT Cloud",
    message_count: 0,
    created_at: Date.now() / 1000,
    updated_at: Date.now() / 1000,
  },
  {
    id: "conv-rag-demo",
    title: "Document Retrieval & RAG QA",
    message_count: 0,
    created_at: (Date.now() - 3600000) / 1000,
    updated_at: (Date.now() - 3600000) / 1000,
  },
];

export function useConversations(activeId: string | null, onSelect: (id: string) => void) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const activeIdRef = useRef(activeId);
  useEffect(() => {
    activeIdRef.current = activeId;
  }, [activeId]);

  const loadAll = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchConversations();
      if (data && data.length > 0) {
        setConversations(data);
        if (!activeIdRef.current) onSelect(data[0].id);
      } else {
        setConversations([]);
      }
    } catch {
      // Offline fallback: load from localStorage or default items
      if (typeof window !== "undefined") {
        const cached = localStorage.getItem("localgpt_conversations");
        if (cached) {
          try {
            const parsed = JSON.parse(cached);
            setConversations(parsed);
            if (!activeIdRef.current && parsed.length > 0) onSelect(parsed[0].id);
            return;
          } catch {
            // ignore
          }
        }
        // Initialize default list
        setConversations(DEFAULT_CONVERSATIONS);
        if (!activeIdRef.current) onSelect(DEFAULT_CONVERSATIONS[0].id);
      }
    } finally {
      setIsLoading(false);
    }
  }, [onSelect]);

  useEffect(() => {
    loadAll();
  }, [loadAll]);


  // Sync to local storage
  useEffect(() => {
    if (conversations.length > 0 && typeof window !== "undefined") {
      localStorage.setItem("localgpt_conversations", JSON.stringify(conversations));
    }
  }, [conversations]);

  const handleCreate = async () => {
    const newId = `conv_${Date.now()}`;
    const newConv: ConversationSummary = {
      id: newId,
      title: "New Chat",
      message_count: 0,
      created_at: Date.now() / 1000,
      updated_at: Date.now() / 1000,
    };

    try {
      const created = await createConversation("New Chat");
      setConversations((prev) => [created, ...prev]);
      onSelect(created.id);
    } catch {
      // Local fallback
      setConversations((prev) => [newConv, ...prev]);
      onSelect(newConv.id);
    }
  };

  const handleRename = async (id: string, newTitle: string) => {
    try {
      await renameConversation(id, newTitle);
      setConversations((prev) =>
        prev.map((c) => (c.id === id ? { ...c, title: newTitle } : c))
      );
    } catch {
      // Local fallback
      setConversations((prev) =>
        prev.map((c) => (c.id === id ? { ...c, title: newTitle } : c))
      );
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteConversation(id);
    } catch {
      // Local fallback
    }

    const remaining = conversations.filter((c) => c.id !== id);
    setConversations(remaining);
    if (typeof window !== "undefined") {
      localStorage.removeItem(`localgpt_conv_${id}`);
    }

    if (activeId === id) {
      if (remaining.length > 0) {
        onSelect(remaining[0].id);
      } else {
        // Create an empty fresh conversation if all deleted
        const freshId = `conv_${Date.now()}`;
        const freshConv: ConversationSummary = {
          id: freshId,
          title: "New Chat",
          message_count: 0,
          created_at: Date.now() / 1000,
          updated_at: Date.now() / 1000,
        };
        setConversations([freshConv]);
        onSelect(freshId);
      }
    }
  };

  return {
    conversations,
    isLoading,
    error,
    refresh: loadAll,
    createChat: handleCreate,
    renameChat: handleRename,
    deleteChat: handleDelete,
  };
}
