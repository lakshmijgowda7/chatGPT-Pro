"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Sparkles, Mail, Lock, User, Eye, EyeOff, ArrowRight, AlertCircle, CheckCircle2 } from "lucide-react";
import { useAuth } from "../../lib/AuthContext";
import { fetchGoogleAuthUrl } from "../../lib/api";

export default function RegisterPage() {
  const router = useRouter();
  const { register } = useAuth();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);

  const isPasswordLongEnough = password.length >= 8;
  const doPasswordsMatch = password.length > 0 && password === confirmPassword;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!email || !password) {
      setError("Please fill in all required fields.");
      return;
    }

    if (password.length < 8) {
      setError("Password must be at least 8 characters long.");
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setIsSubmitting(true);
    try {
      await register({
        email: email.trim(),
        password,
        name: name.trim() || email.split("@")[0],
        full_name: name.trim() || undefined,
      });
      router.push("/");
    } catch (err: any) {
      setError(err.message || "Registration failed. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleGoogleLogin = async () => {
    setError(null);
    setIsGoogleLoading(true);
    try {
      const authUrl = await fetchGoogleAuthUrl();
      window.location.href = authUrl;
    } catch (err: any) {
      setError(err.message || "Google OAuth is not configured. Please register with email.");
      setIsGoogleLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-screen bg-[#111111] text-gray-100 flex flex-col justify-center items-center p-4 selection:bg-emerald-500 selection:text-white relative overflow-hidden">
      {/* Background Decorative Glow */}
      <div className="absolute -top-40 -right-40 w-96 h-96 bg-emerald-600/15 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-teal-600/15 rounded-full blur-3xl pointer-events-none" />

      {/* Main Container */}
      <div className="w-full max-w-md bg-[#1c1c1c]/90 backdrop-blur-xl border border-gray-800 rounded-3xl p-8 shadow-2xl shadow-black/80 relative z-10 animate-in fade-in zoom-in-95 duration-200">
        {/* Brand Header */}
        <div className="text-center mb-5">
          <div className="inline-flex p-3 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-teal-500/10 border border-emerald-500/30 text-emerald-400 mb-3 shadow-lg shadow-emerald-950/50">
            <Sparkles size={28} />
          </div>
          <h1 className="text-2xl font-bold text-gray-100 tracking-tight">Create an Account</h1>
          <p className="text-sm text-gray-400 mt-1">Get started with LocalGPT AI platform</p>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-4 p-3.5 rounded-xl bg-red-950/60 border border-red-800/60 text-red-300 text-xs flex items-start gap-2.5 animate-in fade-in duration-150">
            <AlertCircle size={16} className="shrink-0 mt-0.5 text-red-400" />
            <span>{error}</span>
          </div>
        )}

        {/* Google OAuth Button */}
        <button
          type="button"
          onClick={handleGoogleLogin}
          disabled={isGoogleLoading || isSubmitting}
          className="w-full py-2.5 px-4 bg-[#232323] hover:bg-[#2a2a2a] border border-gray-700/80 hover:border-gray-600 text-gray-200 font-medium text-xs rounded-xl shadow-sm transition-all flex items-center justify-center gap-3 active:scale-[0.99] disabled:opacity-50"
        >
          {isGoogleLoading ? (
            <div className="w-4 h-4 border-2 border-emerald-400/40 border-t-emerald-400 rounded-full animate-spin" />
          ) : (
            <svg className="w-4 h-4" viewBox="0 0 24 24">
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
          )}
          <span>Continue with Google</span>
        </button>

        {/* Divider */}
        <div className="relative my-4">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-gray-800" />
          </div>
          <div className="relative flex justify-center text-[11px] uppercase">
            <span className="bg-[#1c1c1c] px-3 text-gray-500 font-medium">Or register with email</span>
          </div>
        </div>

        {/* Registration Form */}
        <form onSubmit={handleSubmit} className="space-y-3.5">
          {/* Name Field */}
          <div>
            <label className="block text-xs font-semibold text-gray-300 mb-1.5">
              Full Name (Optional)
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-gray-500">
                <User size={16} />
              </div>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Jane Developer"
                className="w-full pl-10 pr-4 py-2 bg-[#141414] border border-gray-800 focus:border-emerald-500/80 focus:ring-2 focus:ring-emerald-500/20 rounded-xl text-sm text-gray-100 placeholder-gray-600 outline-none transition-all"
              />
            </div>
          </div>

          {/* Email Field */}
          <div>
            <label className="block text-xs font-semibold text-gray-300 mb-1.5">
              Email Address <span className="text-emerald-400">*</span>
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-gray-500">
                <Mail size={16} />
              </div>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@example.com"
                className="w-full pl-10 pr-4 py-2 bg-[#141414] border border-gray-800 focus:border-emerald-500/80 focus:ring-2 focus:ring-emerald-500/20 rounded-xl text-sm text-gray-100 placeholder-gray-600 outline-none transition-all"
              />
            </div>
          </div>

          {/* Password Field */}
          <div>
            <label className="block text-xs font-semibold text-gray-300 mb-1.5">
              Password <span className="text-emerald-400">*</span>
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-gray-500">
                <Lock size={16} />
              </div>
              <input
                type={showPassword ? "text" : "password"}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Minimum 8 characters"
                className="w-full pl-10 pr-10 py-2 bg-[#141414] border border-gray-800 focus:border-emerald-500/80 focus:ring-2 focus:ring-emerald-500/20 rounded-xl text-sm text-gray-100 placeholder-gray-600 outline-none transition-all"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-gray-500 hover:text-gray-300"
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          {/* Confirm Password Field */}
          <div>
            <label className="block text-xs font-semibold text-gray-300 mb-1.5">
              Confirm Password <span className="text-emerald-400">*</span>
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-gray-500">
                <Lock size={16} />
              </div>
              <input
                type={showPassword ? "text" : "password"}
                required
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Repeat password"
                className="w-full pl-10 pr-10 py-2 bg-[#141414] border border-gray-800 focus:border-emerald-500/80 focus:ring-2 focus:ring-emerald-500/20 rounded-xl text-sm text-gray-100 placeholder-gray-600 outline-none transition-all"
              />
            </div>
          </div>

          {/* Validation Checklist */}
          <div className="p-2.5 bg-[#141414] border border-gray-800/80 rounded-xl space-y-1 text-[11px]">
            <div className={`flex items-center gap-2 ${isPasswordLongEnough ? "text-emerald-400" : "text-gray-500"}`}>
              <CheckCircle2 size={13} className={isPasswordLongEnough ? "text-emerald-400" : "text-gray-600"} />
              <span>At least 8 characters in length</span>
            </div>
            <div className={`flex items-center gap-2 ${doPasswordsMatch ? "text-emerald-400" : "text-gray-500"}`}>
              <CheckCircle2 size={13} className={doPasswordsMatch ? "text-emerald-400" : "text-gray-600"} />
              <span>Passwords match</span>
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isSubmitting || !isPasswordLongEnough || !doPasswordsMatch}
            className="w-full mt-2 py-2.5 px-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-semibold text-sm rounded-xl shadow-lg shadow-emerald-950/50 transition-all flex items-center justify-center gap-2 active:scale-[0.99] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSubmitting ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <>
                <span>Create Account</span>
                <ArrowRight size={16} />
              </>
            )}
          </button>
        </form>

        {/* Navigation to Login */}
        <div className="mt-5 text-center border-t border-gray-800/80 pt-4">
          <p className="text-xs text-gray-400">
            Already have an account?{" "}
            <Link
              href="/login"
              className="text-emerald-400 hover:text-emerald-300 font-semibold underline underline-offset-4 decoration-emerald-500/40 hover:decoration-emerald-400 transition-colors"
            >
              Sign in
            </Link>
          </p>
          <div className="mt-2">
            <Link
              href="/"
              className="text-xs text-gray-500 hover:text-gray-400 transition-colors"
            >
              ← Back to Chat (Guest Mode)
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
