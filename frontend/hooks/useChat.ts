"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { ChatMessage, SourceReference } from "../types/chat";
import { fetchConversation, getAuthHeaders } from "../lib/api";
import { API_BASE_URL } from "../lib/constants";
import { useAuth } from "../lib/AuthContext";

export const GUEST_MAX_CHATS = 5;

export function useChat(
  conversationId: string | null,
  onConversationCreated?: (newConvId: string) => void,
  onStreamComplete?: () => void
) {
  const { user } = useAuth();
  // A user is ONLY a guest if explicitly browsing via guest pass.
  // Anyone signing in with Google or Email has Unlimited Chats.
  const isGuest = Boolean(
    user &&
    (user.profile?.is_anonymous === true ||
     user.email?.includes("@localgpt.user") ||
     user.email?.startsWith("guest_") ||
     user.id?.startsWith("guest_")) &&
    !user.email?.includes("@gmail.com") &&
    user.profile?.provider !== "google" &&
    user.profile?.provider !== "password"
  );

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [guestChatCount, setGuestChatCount] = useState<number>(0);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Sync guest chat count from local storage
  useEffect(() => {
    if (typeof window !== "undefined") {
      if (!isGuest) {
        setGuestChatCount(0);
        return;
      }
      const saved = parseInt(localStorage.getItem("localgpt_guest_chat_count") || "0", 10);
      setGuestChatCount(saved);
    }
  }, [isGuest]);

  const guestChatsRemaining = isGuest ? Math.max(0, GUEST_MAX_CHATS - guestChatCount) : Infinity;

  // Load message history from API / localStorage when conversation changes
  useEffect(() => {
    if (!conversationId) {
      setMessages([]);
      return;
    }

    let isMounted = true;
    setIsLoadingHistory(true);
    setError(null);

    // Try fetching from backend API
    fetchConversation(conversationId)
      .then((data) => {
        if (isMounted) {
          setMessages(data.messages || []);
        }
      })
      .catch(() => {
        // Fallback to local storage if backend is offline
        if (typeof window !== "undefined") {
          const cached = localStorage.getItem(`localgpt_conv_${conversationId}`);
          if (cached && isMounted) {
            try {
              setMessages(JSON.parse(cached));
            } catch {
              setMessages([]);
            }
          }
        }
      })
      .finally(() => {
        if (isMounted) setIsLoadingHistory(false);
      });

    return () => {
      isMounted = false;
    };
  }, [conversationId]);

  // Persist messages to local storage as fallback
  useEffect(() => {
    if (conversationId && messages.length > 0 && typeof window !== "undefined") {
      localStorage.setItem(`localgpt_conv_${conversationId}`, JSON.stringify(messages));
    }
  }, [conversationId, messages]);

  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const sendMessage = useCallback(
    async (text: string, mode: "chat" | "rag" = "chat") => {
      if (!text.trim() || isStreaming) return;

      // Enforce 5-chat limit for guest users
      if (isGuest && guestChatCount >= GUEST_MAX_CHATS) {
        const limitMsg: ChatMessage = {
          id: `limit_${Date.now()}`,
          conversation_id: conversationId || "",
          role: "assistant",
          content: `Guest Chat Limit Reached (${GUEST_MAX_CHATS}/${GUEST_MAX_CHATS} chats used)\n\nYou have used all 5 of your free guest chats on ChatGPT Pro.\n\nUnlock Unlimited Access: Please sign in with Google or Email to continue chatting with unlimited fast responses, persistent memory, and document intelligence!`,
          created_at: Date.now() / 1000,
        };
        setMessages((prev) => [...prev, limitMsg]);
        setError(`Guest limit reached (${GUEST_MAX_CHATS}/${GUEST_MAX_CHATS}). Please sign in.`);
        return;
      }

      const userMsgId = `usr_${Date.now()}`;
      const userMessage: ChatMessage = {
        id: userMsgId,
        conversation_id: conversationId || "",
        role: "user",
        content: text.trim(),
        created_at: Date.now() / 1000,
      };

      const assistantMsgId = `ast_${Date.now()}`;
      const assistantPlaceholder: ChatMessage = {
        id: assistantMsgId,
        conversation_id: conversationId || "",
        role: "assistant",
        content: "",
        created_at: Date.now() / 1000,
      };

      setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
      setIsStreaming(true);
      setError(null);

      const controller = new AbortController();
      abortControllerRef.current = controller;

      try {
        const authHeaders = getAuthHeaders();
        const response = await fetch(`${API_BASE_URL}/chat/stream`, {
          method: "POST",
          headers: {
            ...authHeaders,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            conversation_id: conversationId,
            message: text.trim(),
            mode: mode,
          }),
          signal: controller.signal,
        });

        if (!response.ok) {
          const errText = await response.text().catch(() => "");
          let errDetail = `Server returned status ${response.status}`;
          try {
            const parsed = JSON.parse(errText);
            if (parsed.detail) errDetail = parsed.detail;
          } catch {}
          throw new Error(errDetail);
        }

        // Increment guest chat count on successful request
        if (isGuest) {
          setGuestChatCount((prev) => {
            const next = prev + 1;
            if (typeof window !== "undefined") {
              localStorage.setItem("localgpt_guest_chat_count", String(next));
            }
            return next;
          });
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder("utf-8");
        let accumulatedText = "";
        let retrievedSources: SourceReference[] | undefined = undefined;

        if (reader) {
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split("\n");

            for (const line of lines) {
              if (line.startsWith("data: ")) {
                try {
                  const data = JSON.parse(line.substring(6));
                  if (data.type === "start") {
                    if (data.sources) {
                      retrievedSources = data.sources;
                    }
                    if (data.conversation_id && onConversationCreated) {
                      onConversationCreated(data.conversation_id);
                    }
                  } else if (data.type === "token") {
                    const cleanToken = (data.token || "").replace(/[\*#]/g, "");
                    accumulatedText += cleanToken;
                    setMessages((prev) =>
                      prev.map((m) =>
                        m.id === assistantMsgId
                          ? { ...m, content: accumulatedText, sources: retrievedSources }
                          : m
                      )
                    );
                  } else if (data.type === "done") {
                    const cleanFinal = (data.content || accumulatedText).replace(/[\*#]/g, "");
                    setMessages((prev) =>
                      prev.map((m) =>
                        m.id === assistantMsgId
                          ? { ...m, id: data.message_id || assistantMsgId, content: cleanFinal, sources: retrievedSources }
                          : m
                      )
                    );
                    if (onStreamComplete) {
                      onStreamComplete();
                    }
                  } else if (data.type === "error") {
                    setError(data.error);
                  }
                } catch {
                  // Skip frame parsing error
                }
              }
            }
          }
        }
      } catch (err: any) {
        if (err.name === "AbortError") {
          return;
        }

        console.error("Backend API connection error:", err);
        setError(err.message || "Failed to reach backend server");
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? {
                  ...m,
                  content: `⚠️ **Connection Error**: Unable to reach backend server at \`${API_BASE_URL}\`.\n\nPlease verify that the FastAPI backend server is running on port 8000 (\`uvicorn app.main:app --port 8000\`).\n\n*Error details: ${err.message || "Network request failed"}*`,
                }
              : m
          )
        );
      } finally {
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    },
    [conversationId, isStreaming, isGuest, guestChatCount, onConversationCreated, onStreamComplete]
  );

  const setFeedback = useCallback((messageId: string, feedback: 'like' | 'dislike') => {
    setMessages((prev) =>
      prev.map((m) => {
        if (m.id === messageId) {
          const newFeedback = m.feedback === feedback ? undefined : feedback;
          return { ...m, feedback: newFeedback };
        }
        return m;
      })
    );
  }, []);

  const editAndResend = useCallback(
    async (messageId: string, newText: string, mode: "chat" | "rag" = "chat") => {
      if (!newText.trim() || isStreaming) return;

      const msgIndex = messages.findIndex((m) => m.id === messageId);
      if (msgIndex === -1) {
        return sendMessage(newText, mode);
      }

      const remainingMessages = messages.slice(0, msgIndex);
      setMessages(remainingMessages);

      return sendMessage(newText, mode);
    },
    [messages, isStreaming, sendMessage]
  );

  const regenerate = useCallback(
    async (mode: "chat" | "rag" = "chat") => {
      if (isStreaming || messages.length === 0) return;

      let lastUserMsgIndex = -1;
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === "user") {
          lastUserMsgIndex = i;
          break;
        }
      }

      if (lastUserMsgIndex === -1) return;

      const lastUserPrompt = messages[lastUserMsgIndex].content;
      const retained = messages.slice(0, lastUserMsgIndex);
      setMessages(retained);

      return sendMessage(lastUserPrompt, mode);
    },
    [messages, isStreaming, sendMessage]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    if (conversationId && typeof window !== "undefined") {
      localStorage.removeItem(`localgpt_conv_${conversationId}`);
    }
  }, [conversationId]);

  return {
    messages,
    isStreaming,
    isLoadingHistory,
    error,
    isGuest,
    guestChatCount,
    guestChatsRemaining,
    sendMessage,
    stopGeneration,
    regenerate,
    editAndResend,
    setFeedback,
    clearMessages,
  };
}
