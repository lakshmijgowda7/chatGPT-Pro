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
  signInWithGoogle: () => Promise<void>;
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
    try {
      const token = getAuthToken();
      if (!token) {
        // Check for cached guest/firebase user in local storage
        if (typeof window !== "undefined") {
          const cachedUser = localStorage.getItem("localgpt_cached_user");
          if (cachedUser) {
            setUser(JSON.parse(cachedUser));
            setIsLoading(false);
            return;
          }
        }
        setUser(null);
        setIsLoading(false);
        return;
      }
      const userData = await getCurrentUser();
      setUser(userData);
      if (typeof window !== "undefined") {
        localStorage.setItem("localgpt_cached_user", JSON.stringify(userData));
      }
    } catch {
      // If token verification with backend fails (e.g. backend offline or guest token)
      if (typeof window !== "undefined") {
        const cachedUser = localStorage.getItem("localgpt_cached_user");
        if (cachedUser) {
          try {
            setUser(JSON.parse(cachedUser));
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
    }
    // Attempt backend sync
    try {
      const backendUser = await getCurrentUser();
      setUser(backendUser);
      if (typeof window !== "undefined") {
        localStorage.setItem("localgpt_cached_user", JSON.stringify(backendUser));
      }
    } catch {
      // Backend is optional/local fallback
    }
  }, []);

  const signInWithGoogle = async () => {
    setIsLoading(true);
    try {
      const res = await fbSignInWithGoogle();
      await handleFirebaseSuccess(res);
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
