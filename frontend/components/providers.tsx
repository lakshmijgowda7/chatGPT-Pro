"use client";

import React, { ReactNode } from "react";
import { AuthProvider } from "../lib/AuthContext";

export function Providers({ children }: { children: ReactNode }) {
  return <AuthProvider>{children}</AuthProvider>;
}
