"use client";

import React, { useState } from "react";
import Link from "next/link";
import { ConversationSummary } from "../../types/chat";
import { PlatformSettings } from "../../types/api";
import { ConversationList } from "./ConversationList";
import { SearchConversations } from "./SearchConversations";
import { SettingsModal } from "./SettingsModal";
import { Plus, Settings, Sparkles, X, LogIn, LogOut, UserPlus, UserCheck, Flame } from "lucide-react";
import { Avatar } from "../ui/Avatar";
import { useAuth } from "../../lib/AuthContext";
import { AuthModal } from "../auth/AuthModal";

interface SidebarProps {
  conversations: ConversationSummary[];
  activeId: string | null;
  isOpenMobile?: boolean;
  onCloseMobile?: () => void;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onRename: (id: string, newTitle: string) => void;
  onDelete: (id: string) => void;
  settings: PlatformSettings | null;
}

export const Sidebar: React.FC<SidebarProps> = ({
  conversations,
  activeId,
  isOpenMobile = false,
  onCloseMobile,
  onSelect,
  onCreate,
  onRename,
  onDelete,
  settings,
}) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const { user, isAuthenticated, logout, signInWithGoogle } = useAuth();

  const filteredCount = conversations.filter((c) =>
    c.title.toLowerCase().includes(searchQuery.toLowerCase().trim())
  ).length;

  const sidebarContent = (
    <div className="flex flex-col h-full bg-[#171717] border-r border-gray-800 select-none">
      {/* Top Header: Brand & Mobile Close */}
      <div className="p-3.5 border-b border-gray-800/80 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-emerald-500 via-teal-400 to-cyan-500 p-0.5 flex items-center justify-center shadow-md shadow-emerald-500/20">
            <div className="w-full h-full bg-[#1e1e1e] rounded-[10px] flex items-center justify-center text-emerald-400">
              <Sparkles size={16} />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-sm text-gray-100">ChatGPT</span>
              <span className="text-[9px] font-black px-1.5 py-0.5 rounded bg-gradient-to-r from-amber-400 to-orange-500 text-black">
                PRO
              </span>
            </div>
            <p className="text-[10px] text-gray-400 leading-tight">Firebase Cloud Platform</p>
          </div>
        </div>

        {/* Mobile close button */}
        {onCloseMobile && (
          <button
            onClick={onCloseMobile}
            className="md:hidden p-1.5 text-gray-400 hover:text-white rounded-lg hover:bg-gray-800"
            aria-label="Close sidebar"
          >
            <X size={18} />
          </button>
        )}
      </div>

      {/* New Chat Button */}
      <div className="p-3 pb-1">
        <button
          onClick={() => {
            onCreate();
            if (onCloseMobile) onCloseMobile();
          }}
          className="w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-md shadow-emerald-950/40 transition-all active:scale-[0.99]"
        >
          <span className="flex items-center gap-2">
            <Plus size={16} />
            <span>New Chat</span>
          </span>
          <span className="text-[10px] bg-white/20 px-1.5 py-0.5 rounded text-white/90">
            Ctrl + N
          </span>
        </button>
      </div>

      {/* Search Conversations */}
      <SearchConversations
        query={searchQuery}
        onQueryChange={setSearchQuery}
        resultCount={filteredCount}
      />

      {/* Conversation List */}
      <ConversationList
        conversations={conversations}
        activeId={activeId}
        searchQuery={searchQuery}
        onSelect={(id) => {
          onSelect(id);
          if (onCloseMobile) onCloseMobile();
        }}
        onRename={onRename}
        onDelete={onDelete}
        onCreateNew={onCreate}
      />

      {/* User Profile & Auth Footer */}
      <div className="p-3 border-t border-gray-800/80 bg-[#141414]">
        {isAuthenticated && user ? (
          /* Authenticated User Profile Card */
          <div className="p-2 rounded-xl bg-[#1e1e1e] border border-gray-800/80">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5 min-w-0">
                <Avatar role="user" size="sm" isOnline={true} />
                <div className="min-w-0">
                  <div className="flex items-center gap-1">
                    <span className="text-xs font-semibold text-gray-200 truncate">
                      {user.name || user.email.split("@")[0]}
                    </span>
                    <span className={`text-[8px] font-black px-1 py-0.2 rounded ${
                      user.profile?.is_anonymous
                        ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                        : "bg-gradient-to-r from-amber-400 to-orange-500 text-black"
                    }`}>
                      {user.profile?.is_anonymous ? "GUEST" : "PRO"}
                    </span>
                  </div>
                  <div className="text-[10px] text-gray-400 truncate">
                    {user.email}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-1">
                <button
                  onClick={() => setIsSettingsOpen(true)}
                  className="p-1.5 rounded-lg text-gray-400 hover:text-gray-100 hover:bg-gray-700/60 transition-colors"
                  title="Settings & System Config"
                >
                  <Settings size={15} />
                </button>
                <button
                  onClick={() => logout()}
                  className="p-1.5 rounded-lg text-gray-400 hover:text-red-400 hover:bg-red-950/40 transition-colors"
                  title="Sign Out"
                >
                  <LogOut size={15} />
                </button>
              </div>
            </div>

            {user.profile?.is_anonymous && (
              <button
                onClick={() => signInWithGoogle()}
                className="mt-2 w-full flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-lg bg-gradient-to-r from-emerald-600 to-teal-600 text-white font-bold text-[11px] hover:opacity-95 transition-all shadow"
              >
                <Sparkles size={12} />
                <span>Switch to Google Account</span>
              </button>
            )}
          </div>
        ) : (
          /* Guest / Logged Out Controls */
          <div className="space-y-2">
            <button
              onClick={() => setIsAuthOpen(true)}
              className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-xl bg-gradient-to-r from-amber-500/20 via-orange-500/20 to-emerald-500/20 hover:from-amber-500/30 hover:to-emerald-500/30 border border-amber-500/30 text-amber-300 font-semibold text-xs transition-all shadow-sm group"
            >
              <Flame size={14} className="text-amber-400 group-hover:scale-110 transition-transform" />
              <span>Sign In / Connect</span>
            </button>
            <div className="flex items-center justify-between px-1 text-[11px] text-gray-500">
              <span className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                <span>Guest Mode Active</span>
              </span>
              <button
                onClick={() => setIsSettingsOpen(true)}
                className="text-gray-400 hover:text-gray-200"
              >
                Settings
              </button>
            </div>
          </div>
        )}
      </div>

      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
      />

      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
      />
    </div>
  );

  return (
    <>
      {/* Desktop Persistent Sidebar */}
      <aside className="hidden md:block w-72 h-full shrink-0">
        {sidebarContent}
      </aside>

      {/* Mobile Drawer Overlay */}
      {isOpenMobile && (
        <div className="fixed inset-0 z-40 md:hidden flex">
          <div
            className="fixed inset-0 bg-black/75 backdrop-blur-sm transition-opacity"
            onClick={onCloseMobile}
          />
          <aside className="relative w-72 h-full z-50 animate-in slide-in-from-left duration-200">
            {sidebarContent}
          </aside>
        </div>
      )}
    </>
  );
};
