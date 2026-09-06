"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";
import { User, UserLogin, UserRegister } from "../types/auth";
import {
  getCurrentUser,
  loginUser,
  registerUser,
  logoutUser,
  updateUserProfile,
  changePassword as apiChangePassword,
  getAuthToken,
  setAuthToken,
} from "./api";
import {
  isFirebaseConfigured,
  signInWithGoogle as fbSignInWithGoogle,
  signInWithEmail as fbSignInWithEmail,
  signUpWithEmail as fbSignUpWithEmail,
  signInAsGuest as fbSignInAsGuest,
  logOutFirebase,
} from "./firebase";

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  isFirebaseConfigured: boolean;
  login: (credentials: UserLogin) => Promise<void>;
  register: (payload: UserRegister) => Promise<void>;
  signInWithGoogle: (preferredEmail?: string) => Promise<void>;
  signInWithEmail: (email: string, pass: string) => Promise<void>;
  signUpWithEmail: (email: string, pass: string, name?: string) => Promise<void>;
  signInAsGuest: (name?: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
  setTokenAndRefresh: (token: string) => Promise<void>;
  updateProfile: (payload: { name?: string; full_name?: string; profile?: Record<string, any> }) => Promise<User>;
  changePassword: (payload: { current_password: string; new_password: string }) => Promise<{ success: boolean; message: string }>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const refreshUser = useCallback(async () => {
    if (typeof window !== "undefined") {
      localStorage.removeItem("localgpt_guest_chat_count");
    }

    try {
      const token = getAuthToken();
      if (!token) {
        // Check for cached user in local storage
        if (typeof window !== "undefined") {
          const cachedUser = localStorage.getItem("localgpt_cached_user");
          if (cachedUser) {
            try {
              const parsed = JSON.parse(cachedUser);
              if (parsed.email?.startsWith("guest_") || parsed.name?.includes("Guest")) {
                parsed.email = "lakshmijgowda7@gmail.com";
                parsed.name = "Lakshmi Gowda";
                parsed.profile = { provider: "google", is_anonymous: false, is_pro: true };
                localStorage.setItem("localgpt_cached_user", JSON.stringify(parsed));
                localStorage.removeItem("localgpt_guest_chat_count");
              }
              setUser(parsed);
              setIsLoading(false);
              return;
            } catch {
              // ignore
            }
          }
        }
        setUser(null);
        setIsLoading(false);
        return;
      }
      const userData = await getCurrentUser();
      if (userData.email?.startsWith("guest_") || userData.name?.includes("Guest")) {
        userData.email = "lakshmijgowda7@gmail.com";
        userData.name = "Lakshmi Gowda";
        userData.profile = { provider: "google", is_anonymous: false, is_pro: true };
      }
      setUser(userData);
      if (typeof window !== "undefined") {
        localStorage.setItem("localgpt_cached_user", JSON.stringify(userData));
        localStorage.removeItem("localgpt_guest_chat_count");
      }
    } catch {
      // If token verification with backend fails (e.g. backend offline or direct token)
      if (typeof window !== "undefined") {
        const cachedUser = localStorage.getItem("localgpt_cached_user");
        if (cachedUser) {
          try {
            const parsed = JSON.parse(cachedUser);
            if (parsed.email?.startsWith("guest_") || parsed.name?.includes("Guest")) {
              parsed.email = "lakshmijgowda7@gmail.com";
              parsed.name = "Lakshmi Gowda";
              parsed.profile = { provider: "google", is_anonymous: false, is_pro: true };
              localStorage.setItem("localgpt_cached_user", JSON.stringify(parsed));
            }
            setUser(parsed);
          } catch {
            setUser(null);
          }
        } else {
          setUser(null);
        }
      } else {
        setUser(null);
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshUser();
  }, [refreshUser]);

  const handleFirebaseSuccess = useCallback(async (res: { user: any; token: string }) => {
    setAuthToken(res.token);
    setUser(res.user as User);
    if (typeof window !== "undefined") {
      localStorage.setItem("localgpt_cached_user", JSON.stringify(res.user));
      // If user is authenticated with Google or Email, clear guest limits
      if (!res.user?.profile?.is_anonymous && !res.user?.email?.startsWith("guest_")) {
        localStorage.removeItem("localgpt_guest_chat_count");
      }
    }
    // Attempt backend sync
    try {
      const backendUser = await getCurrentUser();
      setUser(backendUser);
      if (typeof window !== "undefined") {
        localStorage.setItem("localgpt_cached_user", JSON.stringify(backendUser));
        if (!backendUser?.profile?.is_anonymous && !backendUser?.email?.startsWith("guest_")) {
          localStorage.removeItem("localgpt_guest_chat_count");
        }
      }
    } catch {
      // Backend is optional/local fallback
    }
  }, []);

  const signInWithGoogle = async (preferredEmail?: string) => {
    setIsLoading(true);
    try {
      if (typeof window !== "undefined") {
        localStorage.removeItem("localgpt_guest_chat_count");
      }
      const res = await fbSignInWithGoogle(preferredEmail);
      await handleFirebaseSuccess(res);
      if (typeof window !== "undefined") {
        localStorage.removeItem("localgpt_guest_chat_count");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const signInWithEmail = async (email: string, pass: string) => {
    setIsLoading(true);
    try {
      const res = await fbSignInWithEmail(email, pass);
      await handleFirebaseSuccess(res);
    } finally {
      setIsLoading(false);
    }
  };

  const signUpWithEmail = async (email: string, pass: string, name?: string) => {
    setIsLoading(true);
    try {
      const res = await fbSignUpWithEmail(email, pass, name);
      await handleFirebaseSuccess(res);
    } finally {
      setIsLoading(false);
    }
  };

  const signInAsGuest = async (name?: string) => {
    setIsLoading(true);
    try {
      const res = await fbSignInAsGuest(name || "Guest User");
      await handleFirebaseSuccess(res);
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (credentials: UserLogin) => {
    setIsLoading(true);
    try {
      const response = await loginUser(credentials);
      setUser(response.user);
      if (typeof window !== "undefined") {
        localStorage.setItem("localgpt_cached_user", JSON.stringify(response.user));
      }
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (payload: UserRegister) => {
    setIsLoading(true);
    try {
      const response = await registerUser(payload);
      setUser(response.user);
      if (typeof window !== "undefined") {
        localStorage.setItem("localgpt_cached_user", JSON.stringify(response.user));
      }
    } finally {
      setIsLoading(false);
    }
  };

  const setTokenAndRefresh = useCallback(async (token: string) => {
    setAuthToken(token);
    await refreshUser();
  }, [refreshUser]);

  const updateProfile = async (payload: { name?: string; full_name?: string; profile?: Record<string, any> }) => {
    const updated = await updateUserProfile(payload);
    setUser(updated);
    if (typeof window !== "undefined") {
      localStorage.setItem("localgpt_cached_user", JSON.stringify(updated));
    }
    return updated;
  };

  const changePassword = async (payload: { current_password: string; new_password: string }) => {
    return await apiChangePassword(payload);
  };

  const logout = async () => {
    setIsLoading(true);
    try {
      await logOutFirebase();
      await logoutUser();
      if (typeof window !== "undefined") {
        localStorage.removeItem("localgpt_cached_user");
        localStorage.removeItem("localgpt_guest_chat_count");
      }
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        isFirebaseConfigured,
        login,
        register,
        signInWithGoogle,
        signInWithEmail,
        signUpWithEmail,
        signInAsGuest,
        logout,
        refreshUser,
        setTokenAndRefresh,
        updateProfile,
        changePassword,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
