import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "../components/providers";

export const metadata: Metadata = {
  title: "ChatGPT Pro — Intelligence & Document Analysis",
  description: "ChatGPT Pro conversational AI assistant with hosted real-time LLM inference, Firebase authentication, and document RAG.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased bg-[#111111] text-gray-100">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
