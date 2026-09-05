"use client";

import React, { useState } from "react";
import { useAuth } from "../../lib/AuthContext";
import {
  Sparkles,
  Mail,
  Lock,
  User as UserIcon,
  ArrowRight,
  ShieldCheck,
  AlertCircle,
  Loader2,
  Zap,
} from "lucide-react";

export const AuthGate: React.FC = () => {
  const {
    signInWithGoogle,
    signInWithEmail,
    signUpWithEmail,
    signInAsGuest,
    isFirebaseConfigured,
  } = useAuth();

  const [tab, setTab] = useState<"quick" | "email">("quick");
  const [isSignUp, setIsSignUp] = useState(false);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGoogleSignIn = async () => {
    setLoading(true);
    setError(null);
    try {
      await signInWithGoogle();
    } catch (err: any) {
      setError(err.message || "Failed to sign in with Google.");
    } finally {
      setLoading(false);
    }
  };

  const handleGuestSignIn = async () => {
    setLoading(true);
    setError(null);
    try {
      await signInAsGuest("Pro Guest Explorer");
    } catch (err: any) {
      setError(err.message || "Failed to start guest session.");
    } finally {
      setLoading(false);
    }
  };

  const handleEmailAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !password.trim()) {
      setError("Please fill in all required fields.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      if (isSignUp) {
        await signUpWithEmail(email.trim(), password, name.trim() || undefined);
      } else {
        await signInWithEmail(email.trim(), password);
      }
    } catch (err: any) {
      setError(err.message || "Authentication failed. Please verify your credentials.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative flex min-h-screen w-screen items-center justify-center overflow-hidden bg-[#181818] font-sans text-gray-100 selection:bg-emerald-500/30 selection:text-emerald-200">
      {/* Background Radial Glow */}
      <div className="pointer-events-none absolute -top-40 left-1/2 -translate-x-1/2 h-[550px] w-[550px] rounded-full bg-emerald-500/10 blur-[130px]" />
      <div className="pointer-events-none absolute -bottom-40 left-1/2 -translate-x-1/2 h-[450px] w-[450px] rounded-full bg-teal-500/10 blur-[120px]" />

      <div className="relative z-10 w-full max-w-md p-6 sm:p-8 animate-in fade-in zoom-in-95 duration-300">
        {/* ChatGPT Pro Emblem */}
        <div className="flex flex-col items-center text-center mb-8">
          <div className="relative mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-tr from-emerald-500 via-teal-400 to-cyan-500 p-0.5 shadow-2xl shadow-emerald-500/30">
            <div className="flex h-full w-full items-center justify-center rounded-[14px] bg-[#1e1e1e]">
              <Sparkles className="h-8 w-8 text-emerald-400 animate-pulse" />
            </div>
            <span className="absolute -top-1.5 -right-2 flex h-5 items-center px-1.5 rounded-full bg-gradient-to-r from-amber-400 to-orange-500 text-[10px] font-black tracking-wider text-black shadow-md">
              PRO
            </span>
          </div>

          <div className="flex items-center gap-2 mb-1.5">
            <h1 className="text-3xl font-extrabold tracking-tight text-white">
              ChatGPT <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-300">Pro</span>
            </h1>
          </div>
          <p className="text-sm text-gray-400 max-w-xs">
            Log in to unlock high-speed inference, document intelligence, and unlimited AI chats.
          </p>

          {/* Status Badge */}
          <div className="mt-4 inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-950/40 border border-emerald-800/40 text-[11px] font-medium text-emerald-300">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>
              {isFirebaseConfigured
                ? "Secured with Firebase Cloud Auth"
                : "Instant Access Active — Enter freely with 1 click"}
            </span>
          </div>
        </div>

        {/* Card Container */}
        <div className="overflow-hidden rounded-2xl border border-gray-800 bg-[#212121]/90 p-6 shadow-2xl backdrop-blur-xl">
          {/* Navigation Tabs */}
          <div className="flex rounded-xl bg-black/40 p-1 mb-5 border border-white/5">
            <button
              type="button"
              onClick={() => {
                setTab("quick");
                setError(null);
              }}
              className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
                tab === "quick"
                  ? "bg-[#2f2f2f] text-white shadow-sm"
                  : "text-gray-400 hover:text-gray-200"
              }`}
            >
              1-Click / Google
            </button>
            <button
              type="button"
              onClick={() => {
                setTab("email");
                setError(null);
              }}
              className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
                tab === "email"
                  ? "bg-[#2f2f2f] text-white shadow-sm"
                  : "text-gray-400 hover:text-gray-200"
              }`}
            >
              Email & Password
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-4 flex items-start gap-2.5 p-3 rounded-xl bg-red-950/40 border border-red-800/50 text-xs text-red-200 animate-in fade-in">
              <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
              <span className="leading-relaxed">{error}</span>
            </div>
          )}

          {/* Quick Access Tab */}
          {tab === "quick" && (
            <div className="space-y-3.5">
              {/* Google OAuth Button */}
              <button
                onClick={handleGoogleSignIn}
                disabled={loading}
                className="w-full flex items-center justify-center gap-3 py-3 px-4 rounded-xl bg-white text-gray-900 font-bold text-sm hover:bg-gray-100 active:scale-[0.99] transition-all shadow-md disabled:opacity-50"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24">
                  <path
                    fill="#4285F4"
                    d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  />
                  <path
                    fill="#34A853"
                    d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  />
                  <path
                    fill="#FBBC05"
                    d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
                  />
                  <path
                    fill="#EA4335"
                    d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
                  />
                </svg>
                <span>{loading ? "Authenticating..." : "Continue with Google"}</span>
              </button>

              {/* Instant 1-Click Guest Button */}
              <button
                onClick={handleGuestSignIn}
                disabled={loading}
                className="w-full flex items-center justify-between py-3 px-4 rounded-xl border border-gray-700/80 bg-[#2a2a2a] hover:bg-[#333333] hover:border-emerald-500/50 text-gray-200 font-semibold text-sm transition-all group disabled:opacity-50"
              >
                <div className="flex items-center gap-2.5">
                  <Zap className="w-4 h-4 text-emerald-400 group-hover:scale-110 transition-transform" />
                  <span>Try Instant Guest Pass</span>
                </div>
                <span className="text-xs text-emerald-400 flex items-center gap-1 font-bold">
                  Open Chat <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                </span>
              </button>

              <div className="pt-2 text-center">
                <span className="text-[11px] text-gray-500">
                  Guest passes allow immediate access with private, isolated chat sessions.
                </span>
              </div>
            </div>
          )}

          {/* Email / Password Tab */}
          {tab === "email" && (
            <form onSubmit={handleEmailAuth} className="space-y-3.5">
              {isSignUp && (
                <div>
                  <label className="block text-xs font-semibold text-gray-300 mb-1">Your Name</label>
                  <div className="relative">
                    <UserIcon className="absolute left-3.5 top-3 w-4 h-4 text-gray-500" />
                    <input
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="Alex Developer"
                      className="w-full pl-10 pr-3.5 py-2.5 text-sm bg-black/40 border border-gray-700/80 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500 transition-colors"
                    />
                  </div>
                </div>
              )}

              <div>
                <label className="block text-xs font-semibold text-gray-300 mb-1">Email Address</label>
                <div className="relative">
                  <Mail className="absolute left-3.5 top-3 w-4 h-4 text-gray-500" />
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="name@company.com"
                    className="w-full pl-10 pr-3.5 py-2.5 text-sm bg-black/40 border border-gray-700/80 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-gray-300 mb-1">Password</label>
                <div className="relative">
                  <Lock className="absolute left-3.5 top-3 w-4 h-4 text-gray-500" />
                  <input
                    type="password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••••••"
                    className="w-full pl-10 pr-3.5 py-2.5 text-sm bg-black/40 border border-gray-700/80 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500 transition-colors"
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full mt-2 py-3 px-4 rounded-xl bg-gradient-to-r from-emerald-600 via-teal-600 to-emerald-500 text-white font-bold text-sm hover:opacity-95 active:scale-[0.99] transition-all shadow-lg shadow-emerald-950/40 flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {loading && <Loader2 className="w-4 h-4 animate-spin" />}
                <span>{isSignUp ? "Create ChatGPT Pro Account" : "Sign In to ChatGPT Pro"}</span>
              </button>

              <div className="text-center pt-2">
                <button
                  type="button"
                  onClick={() => {
                    setIsSignUp(!isSignUp);
                    setError(null);
                  }}
                  className="text-xs text-gray-400 hover:text-emerald-400 transition-colors"
                >
                  {isSignUp
                    ? "Already have an account? Sign In"
                    : "New to ChatGPT Pro? Create an account"}
                </button>
              </div>
            </form>
          )}
        </div>

        {/* Footer info */}
        <p className="mt-8 text-center text-xs text-gray-500">
          ChatGPT Pro Cloud Platform. Protected by Firebase Authentication.
        </p>
      </div>
    </div>
  );
};
