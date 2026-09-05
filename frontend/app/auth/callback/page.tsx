"use client";

import React, { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Sparkles, AlertCircle, CheckCircle2 } from "lucide-react";
import { useAuth } from "../../../lib/AuthContext";
import Link from "next/link";

function CallbackContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { setTokenAndRefresh } = useAuth();
  const [status, setStatus] = useState<"loading" | "success" | "error">("loading");
  const [errorMessage, setErrorMessage] = useState<string>("");

  useEffect(() => {
    const token = searchParams.get("token");
    const error = searchParams.get("error");

    if (error) {
      setStatus("error");
      setErrorMessage(decodeURIComponent(error));
      return;
    }

    if (token) {
      setTokenAndRefresh(token)
        .then(() => {
          setStatus("success");
          setTimeout(() => {
            router.push("/");
          }, 800);
        })
        .catch((err) => {
          setStatus("error");
          setErrorMessage(err.message || "Failed to authenticate session with Google.");
        });
    } else {
      setStatus("error");
      setErrorMessage("No authentication token found in callback URL.");
    }
  }, [searchParams, router, setTokenAndRefresh]);

  return (
    <div className="w-full max-w-md bg-[#1c1c1c]/90 backdrop-blur-xl border border-gray-800 rounded-3xl p-8 shadow-2xl shadow-black/80 text-center relative z-10 animate-in fade-in zoom-in-95 duration-200">
      <div className="inline-flex p-3.5 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-teal-500/10 border border-emerald-500/30 text-emerald-400 mb-4 shadow-lg shadow-emerald-950/50">
        <Sparkles size={28} />
      </div>

      {status === "loading" && (
        <div className="space-y-3">
          <h2 className="text-xl font-bold text-gray-100">Authenticating with Google...</h2>
          <p className="text-xs text-gray-400">Verifying session credentials and securing your token.</p>
          <div className="pt-4 flex justify-center">
            <div className="w-8 h-8 border-3 border-emerald-500/30 border-t-emerald-400 rounded-full animate-spin" />
          </div>
        </div>
      )}

      {status === "success" && (
        <div className="space-y-3">
          <div className="flex justify-center text-emerald-400">
            <CheckCircle2 size={36} />
          </div>
          <h2 className="text-xl font-bold text-gray-100">Authentication Successful!</h2>
          <p className="text-xs text-gray-400">Redirecting you to your chat workspace...</p>
        </div>
      )}

      {status === "error" && (
        <div className="space-y-4">
          <div className="flex justify-center text-red-400">
            <AlertCircle size={36} />
          </div>
          <h2 className="text-xl font-bold text-gray-100">Authentication Failed</h2>
          <p className="text-xs text-red-300/90 bg-red-950/40 p-3 rounded-xl border border-red-800/40">
            {errorMessage}
          </p>
          <div className="pt-2">
            <Link
              href="/login"
              className="inline-block px-4 py-2 text-xs font-semibold bg-gray-800 hover:bg-gray-700 text-gray-200 rounded-xl transition-colors"
            >
              Return to Login
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}

export default function AuthCallbackPage() {
  return (
    <div className="min-h-screen w-screen bg-[#111111] text-gray-100 flex flex-col justify-center items-center p-4 relative overflow-hidden">
      <div className="absolute -top-40 -left-40 w-96 h-96 bg-emerald-600/15 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -bottom-40 -right-40 w-96 h-96 bg-teal-600/15 rounded-full blur-3xl pointer-events-none" />
      <Suspense fallback={
        <div className="text-center text-gray-400 text-xs">Loading authentication state...</div>
      }>
        <CallbackContent />
      </Suspense>
    </div>
  );
}
