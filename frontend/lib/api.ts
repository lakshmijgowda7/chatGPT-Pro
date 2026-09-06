import { API_BASE_URL } from "./constants";
import { ConversationSummary, ConversationDetail } from "../types/chat";
import { DocumentItem } from "../types/document";
import { PlatformSettings } from "../types/api";
import { User, UserLogin, UserRegister, AuthResponse } from "../types/auth";

const TOKEN_KEY = "localgpt_access_token";

export function getAuthToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function setAuthToken(token: string): void {
  if (typeof window !== "undefined") {
    localStorage.setItem(TOKEN_KEY, token);
  }
}

export function removeAuthToken(): void {
  if (typeof window !== "undefined") {
    localStorage.removeItem(TOKEN_KEY);
  }
}

export function getAuthHeaders(): HeadersInit {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  const token = getAuthToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
}

export async function fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs = 3000): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      ...options,
      signal: options.signal || controller.signal,
    });
    return res;
  } finally {
    clearTimeout(timeoutId);
  }
}

// -----------------------------------------------------------------------------
// Authentication Endpoints
// -----------------------------------------------------------------------------

export async function registerUser(payload: UserRegister): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE_URL}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || data.message || "Registration failed");
  }
  if (data.access_token) {
    setAuthToken(data.access_token);
  }
  return data;
}

export async function loginUser(credentials: UserLogin): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(credentials),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || data.message || "Invalid email or password");
  }
  if (data.access_token) {
    setAuthToken(data.access_token);
  }
  return data;
}

export async function fetchGoogleAuthUrl(): Promise<string> {
  const res = await fetch(`${API_BASE_URL}/auth/google/url`);
  if (!res.ok) {
    // Direct redirect to backend endpoint if url API is not directly used
    return `${API_BASE_URL}/auth/google/login`;
  }
  const data = await res.json();
  return data.url;
}

export async function getCurrentUser(): Promise<User> {
  const token = getAuthToken();
  if (!token) throw new Error("No token available");

  const res = await fetch(`${API_BASE_URL}/auth/me`, {
    headers: getAuthHeaders(),
  });
  const data = await res.json();
  if (!res.ok) {
    if (!token.startsWith("token_") && !token.startsWith("demo_")) {
      removeAuthToken();
    }
    throw new Error(data.detail || data.message || "Session expired");
  }
  return data;
}

export async function updateUserProfile(payload: { name?: string; full_name?: string; profile?: Record<string, any> }): Promise<User> {
  const res = await fetch(`${API_BASE_URL}/auth/me`, {
    method: "PATCH",
    headers: getAuthHeaders(),
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || data.message || "Failed to update profile");
  }
  return data;
}

export async function changePassword(payload: { current_password: string; new_password: string }): Promise<{ success: boolean; message: string }> {
  const res = await fetch(`${API_BASE_URL}/auth/change-password`, {
    method: "POST",
    headers: getAuthHeaders(),
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || data.message || "Failed to change password");
  }
  return data;
}

export async function logoutUser(): Promise<void> {
  try {
    const token = getAuthToken();
    if (token) {
      await fetch(`${API_BASE_URL}/auth/logout`, {
        method: "POST",
        headers: getAuthHeaders(),
      });
    }
  } catch {
    // Ignore network failure on logout
  } finally {
    removeAuthToken();
  }
}


// -----------------------------------------------------------------------------
// Conversation Endpoints (Scoped with Auth)
// -----------------------------------------------------------------------------

export async function fetchConversations(): Promise<ConversationSummary[]> {
  const res = await fetchWithTimeout(`${API_BASE_URL}/conversations`, {
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error("Failed to fetch conversations");
  return res.json();
}

export async function fetchConversation(id: string): Promise<ConversationDetail> {
  const res = await fetchWithTimeout(`${API_BASE_URL}/conversations/${id}`, {
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error("Failed to fetch conversation");
  return res.json();
}

export async function createConversation(title: string = "New Chat"): Promise<ConversationSummary> {
  const res = await fetch(`${API_BASE_URL}/conversations`, {
    method: "POST",
    headers: getAuthHeaders(),
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error("Failed to create conversation");
  return res.json();
}

export async function renameConversation(id: string, title: string): Promise<ConversationSummary> {
  const res = await fetch(`${API_BASE_URL}/conversations/${id}`, {
    method: "PATCH",
    headers: getAuthHeaders(),
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error("Failed to rename conversation");
  return res.json();
}

export async function deleteConversation(id: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/conversations/${id}`, {
    method: "DELETE",
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error("Failed to delete conversation");
}

export async function fetchDocuments(): Promise<DocumentItem[]> {
  const res = await fetchWithTimeout(`${API_BASE_URL}/documents`, {
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error("Failed to fetch documents");
  return res.json();
}

export async function fetchSettings(): Promise<PlatformSettings> {
  const res = await fetchWithTimeout(`${API_BASE_URL}/settings`);
  if (!res.ok) throw new Error("Failed to fetch platform settings");
  return res.json();
}

export async function updateSettings(payload: Partial<PlatformSettings> & { llm_api_key?: string }): Promise<PlatformSettings> {
  const res = await fetch(`${API_BASE_URL}/settings`, {
    method: "PATCH",
    headers: getAuthHeaders(),
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || data.message || "Failed to update platform settings");
  }
  return data;
}

export async function requestPasswordReset(email: string): Promise<{ success: boolean; message: string; reset_token?: string }> {
  const res = await fetch(`${API_BASE_URL}/auth/forgot-password`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || data.message || "Failed to request password reset");
  }
  return data;
}

export async function resetPasswordWithToken(token: string, newPassword: string): Promise<{ success: boolean; message: string }> {
  const res = await fetch(`${API_BASE_URL}/auth/reset-password`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token, new_password: newPassword }),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || data.message || "Failed to reset password");
  }
  return data;
}

