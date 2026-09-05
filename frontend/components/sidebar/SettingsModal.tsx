import React, { useState, useEffect } from "react";
import { PlatformSettings } from "../../types/api";
import {
  Server,
  Cpu,
  Key,
  Thermometer,
  Database,
  Check,
  Sliders,
  User as UserIcon,
  Shield,
  Save,
  AlertCircle,
  Lock,
  Sparkles,
} from "lucide-react";
import { Modal } from "../ui/Modal";
import { Button } from "../ui/Button";
import { API_BASE_URL } from "../../lib/constants";
import { updateSettings } from "../../lib/api";
import { useAuth } from "../../lib/AuthContext";

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: PlatformSettings | null;
}

const PROVIDER_PRESETS: Record<
  string,
  { name: string; baseUrl: string; models: string[]; keyPlaceholder: string }
> = {
  groq: {
    name: "Groq (Ultra-Fast Hosted Cloud)",
    baseUrl: "https://api.groq.com/openai/v1",
    models: ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"],
    keyPlaceholder: "Paste your Groq API key (starts with gsk_...)",
  },
  openai: {
    name: "OpenAI",
    baseUrl: "https://api.openai.com/v1",
    models: ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    keyPlaceholder: "Paste your OpenAI key (starts with sk-...)",
  },
  openrouter: {
    name: "OpenRouter (Multi-Model Hub)",
    baseUrl: "https://openrouter.ai/api/v1",
    models: [
      "meta-llama/llama-3.3-70b-instruct:free",
      "google/gemini-2.0-flash-exp:free",
      "deepseek/deepseek-r1:free",
      "meta-llama/llama-3.1-8b-instruct:free",
    ],
    keyPlaceholder: "Paste your OpenRouter key (starts with sk-or-...)",
  },
  gemini: {
    name: "Google Gemini (OpenAI Endpoint)",
    baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai/",
    models: ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
    keyPlaceholder: "Paste your Gemini API key (starts with AIzaSy...)",
  },
  together: {
    name: "Together AI",
    baseUrl: "https://api.together.xyz/v1",
    models: ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
    keyPlaceholder: "Paste your Together AI API key",
  },
};

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
}) => {
  const { user, isAuthenticated, updateProfile, changePassword } = useAuth();
  const [activeTab, setActiveTab] = useState<"profile" | "security" | "platform">("profile");

  // Profile Edit State
  const [displayName, setDisplayName] = useState(user?.name || "");
  const [isSavingProfile, setIsSavingProfile] = useState(false);
  const [profileSuccess, setProfileSuccess] = useState<string | null>(null);
  const [profileError, setProfileError] = useState<string | null>(null);

  // Password Change State
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isChangingPassword, setIsChangingPassword] = useState(false);
  const [passwordSuccess, setPasswordSuccess] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);

  // Platform & LLM Config State
  const [selectedProvider, setSelectedProvider] = useState(settings?.llm_provider || "groq");
  const [selectedModel, setSelectedModel] = useState(settings?.llm_model || "llama-3.3-70b-versatile");
  const [customBaseUrl, setCustomBaseUrl] = useState(settings?.llm_base_url || "https://api.groq.com/openai/v1");
  const [customApiKey, setCustomApiKey] = useState("");
  const [temperature, setTemperature] = useState(settings?.default_temperature ?? 0.7);
  const [topP, setTopP] = useState(settings?.default_top_p ?? 0.9);
  const [maxTokens, setMaxTokens] = useState(settings?.default_max_tokens ?? 2048);
  const [isSavingLlm, setIsSavingLlm] = useState(false);
  const [llmSuccess, setLlmSuccess] = useState<string | null>(null);
  const [llmError, setLlmError] = useState<string | null>(null);

  // Reset fields on modal open or settings update
  useEffect(() => {
    if (user?.name) {
      setDisplayName(user.name);
    }
  }, [user]);

  useEffect(() => {
    if (settings) {
      setSelectedProvider(settings.llm_provider || "groq");
      setSelectedModel(settings.llm_model || "llama-3.3-70b-versatile");
      setCustomBaseUrl(settings.llm_base_url || "https://api.groq.com/openai/v1");
      setTemperature(settings.default_temperature ?? 0.7);
      setTopP(settings.default_top_p ?? 0.9);
      setMaxTokens(settings.default_max_tokens ?? 2048);
    }
  }, [settings]);

  const handleProviderChange = (providerKey: string) => {
    setSelectedProvider(providerKey);
    const preset = PROVIDER_PRESETS[providerKey];
    if (preset) {
      setCustomBaseUrl(preset.baseUrl);
      setSelectedModel(preset.models[0] || "");
    }
  };

  const handleSaveProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    setProfileError(null);
    setProfileSuccess(null);
    setIsSavingProfile(true);

    try {
      await updateProfile({ name: displayName.trim() });
      setProfileSuccess("Profile updated successfully!");
      setTimeout(() => setProfileSuccess(null), 3000);
    } catch (err: any) {
      setProfileError(err.message || "Failed to update profile.");
    } finally {
      setIsSavingProfile(false);
    }
  };

  const handleChangePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setPasswordError(null);
    setPasswordSuccess(null);

    if (newPassword.length < 8) {
      setPasswordError("New password must be at least 8 characters.");
      return;
    }

    if (newPassword !== confirmPassword) {
      setPasswordError("New passwords do not match.");
      return;
    }

    setIsChangingPassword(true);
    try {
      await changePassword({ current_password: currentPassword, new_password: newPassword });
      setPasswordSuccess("Password changed successfully!");
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      setTimeout(() => setPasswordSuccess(null), 3000);
    } catch (err: any) {
      setPasswordError(err.message || "Failed to change password. Please check your current password.");
    } finally {
      setIsChangingPassword(false);
    }
  };

  const handleSaveLlmSettings = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSavingLlm(true);
    setLlmError(null);
    setLlmSuccess(null);

    try {
      await updateSettings({
        llm_provider: selectedProvider,
        llm_model: selectedModel,
        llm_base_url: customBaseUrl,
        llm_api_key: customApiKey.trim() || undefined,
        default_temperature: temperature,
        default_top_p: topP,
        default_max_tokens: maxTokens,
      });
      setLlmSuccess("LLM inference configuration updated successfully!");
      setCustomApiKey("");
      setTimeout(() => setLlmSuccess(null), 3000);
    } catch (err: any) {
      setLlmError(err.message || "Failed to update LLM configuration.");
    } finally {
      setIsSavingLlm(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={
        <div className="flex items-center gap-2">
          <Sliders size={18} className="text-emerald-400" />
          <span>Platform Settings & Preferences</span>
        </div>
      }
      maxWidth="lg"
    >
      <div className="space-y-4 text-xs text-gray-300">
        {/* Navigation Tabs */}
        <div className="flex border-b border-gray-800 pb-2 gap-2">
          <button
            onClick={() => setActiveTab("profile")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === "profile"
                ? "bg-emerald-950/80 text-emerald-400 border border-emerald-800/60"
                : "text-gray-400 hover:text-gray-200 hover:bg-gray-800/60"
            }`}
          >
            <UserIcon size={14} />
            <span>Profile & Account</span>
          </button>

          <button
            onClick={() => setActiveTab("security")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === "security"
                ? "bg-emerald-950/80 text-emerald-400 border border-emerald-800/60"
                : "text-gray-400 hover:text-gray-200 hover:bg-gray-800/60"
            }`}
          >
            <Shield size={14} />
            <span>Security & Password</span>
          </button>

          <button
            onClick={() => setActiveTab("platform")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === "platform"
                ? "bg-emerald-950/80 text-emerald-400 border border-emerald-800/60"
                : "text-gray-400 hover:text-gray-200 hover:bg-gray-800/60"
            }`}
          >
            <Server size={14} />
            <span>Hosted LLM & Models</span>
          </button>
        </div>

        {/* TAB 1: Profile & Account */}
        {activeTab === "profile" && (
          <div className="space-y-4 animate-in fade-in duration-150">
            {isAuthenticated && user ? (
              <form onSubmit={handleSaveProfile} className="space-y-3.5">
                {profileSuccess && (
                  <div className="p-2.5 rounded-xl bg-emerald-950/60 border border-emerald-800/60 text-emerald-300 text-xs flex items-center gap-2">
                    <Check size={14} className="text-emerald-400" />
                    <span>{profileSuccess}</span>
                  </div>
                )}
                {profileError && (
                  <div className="p-2.5 rounded-xl bg-red-950/60 border border-red-800/60 text-red-300 text-xs flex items-center gap-2">
                    <AlertCircle size={14} className="text-red-400" />
                    <span>{profileError}</span>
                  </div>
                )}

                <div>
                  <label className="block text-[11px] font-semibold text-gray-400 mb-1 uppercase tracking-wider">
                    Display Name
                  </label>
                  <input
                    type="text"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    placeholder="Enter your name"
                    className="w-full bg-[#1e1e1e] border border-gray-800 focus:border-emerald-500 rounded-xl px-3 py-2 text-xs text-gray-100 outline-none"
                  />
                </div>

                <div>
                  <label className="block text-[11px] font-semibold text-gray-400 mb-1 uppercase tracking-wider">
                    Registered Email
                  </label>
                  <input
                    type="email"
                    disabled
                    value={user.email}
                    className="w-full bg-[#181818] border border-gray-800/60 rounded-xl px-3 py-2 text-xs text-gray-500 cursor-not-allowed"
                  />
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px] text-gray-400 pt-1">
                  <div className="p-2.5 rounded-xl bg-[#1a1a1a] border border-gray-800">
                    <span className="text-gray-500 block">Account ID</span>
                    <span className="font-mono text-gray-300 truncate block">{user.id}</span>
                  </div>
                  <div className="p-2.5 rounded-xl bg-[#1a1a1a] border border-gray-800">
                    <span className="text-gray-500 block">Role</span>
                    <span className="font-semibold text-emerald-400 block">
                      {user.is_superuser ? "Administrator" : "Standard User"}
                    </span>
                  </div>
                </div>

                <div className="pt-2 flex justify-end">
                  <Button
                    type="submit"
                    size="sm"
                    disabled={isSavingProfile}
                    className="flex items-center gap-1.5"
                  >
                    {isSavingProfile ? (
                      <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    ) : (
                      <Save size={13} />
                    )}
                    <span>Save Changes</span>
                  </Button>
                </div>
              </form>
            ) : (
              <div className="p-6 text-center text-gray-400 space-y-2">
                <p>
                  You are currently browsing in <strong>Guest Mode</strong>.
                </p>
                <p className="text-[11px] text-gray-500">
                  Sign in to sync your conversation history and manage your account settings.
                </p>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: Security & Password */}
        {activeTab === "security" && (
          <div className="space-y-4 animate-in fade-in duration-150">
            {isAuthenticated ? (
              <form onSubmit={handleChangePassword} className="space-y-3">
                {passwordSuccess && (
                  <div className="p-2.5 rounded-xl bg-emerald-950/60 border border-emerald-800/60 text-emerald-300 text-xs flex items-center gap-2">
                    <Check size={14} className="text-emerald-400" />
                    <span>{passwordSuccess}</span>
                  </div>
                )}
                {passwordError && (
                  <div className="p-2.5 rounded-xl bg-red-950/60 border border-red-800/60 text-red-300 text-xs flex items-center gap-2">
                    <AlertCircle size={14} className="text-red-400" />
                    <span>{passwordError}</span>
                  </div>
                )}

                <div>
                  <label className="block text-[11px] font-semibold text-gray-400 mb-1 uppercase tracking-wider">
                    Current Password
                  </label>
                  <input
                    type="password"
                    required
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full bg-[#1e1e1e] border border-gray-800 focus:border-emerald-500 rounded-xl px-3 py-2 text-xs text-gray-100 outline-none"
                  />
                </div>

                <div>
                  <label className="block text-[11px] font-semibold text-gray-400 mb-1 uppercase tracking-wider">
                    New Password
                  </label>
                  <input
                    type="password"
                    required
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="Minimum 8 characters"
                    className="w-full bg-[#1e1e1e] border border-gray-800 focus:border-emerald-500 rounded-xl px-3 py-2 text-xs text-gray-100 outline-none"
                  />
                </div>

                <div>
                  <label className="block text-[11px] font-semibold text-gray-400 mb-1 uppercase tracking-wider">
                    Confirm New Password
                  </label>
                  <input
                    type="password"
                    required
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="Repeat new password"
                    className="w-full bg-[#1e1e1e] border border-gray-800 focus:border-emerald-500 rounded-xl px-3 py-2 text-xs text-gray-100 outline-none"
                  />
                </div>

                <div className="pt-2 flex justify-end">
                  <Button
                    type="submit"
                    size="sm"
                    disabled={isChangingPassword}
                    className="flex items-center gap-1.5"
                  >
                    {isChangingPassword ? (
                      <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    ) : (
                      <Lock size={13} />
                    )}
                    <span>Update Password</span>
                  </Button>
                </div>
              </form>
            ) : (
              <div className="p-6 text-center text-gray-400">
                Sign in to manage security and update passwords.
              </div>
            )}
          </div>
        )}

        {/* TAB 3: Hosted LLM & Models */}
        {activeTab === "platform" && (
          <form onSubmit={handleSaveLlmSettings} className="space-y-3.5 animate-in fade-in duration-150">
            {llmSuccess && (
              <div className="p-2.5 rounded-xl bg-emerald-950/60 border border-emerald-800/60 text-emerald-300 text-xs flex items-center gap-2">
                <Check size={14} className="text-emerald-400" />
                <span>{llmSuccess}</span>
              </div>
            )}
            {llmError && (
              <div className="p-2.5 rounded-xl bg-red-950/60 border border-red-800/60 text-red-300 text-xs flex items-center gap-2">
                <AlertCircle size={14} className="text-red-400" />
                <span>{llmError}</span>
              </div>
            )}

            {/* Provider Selection */}
            <div>
              <label className="block text-[11px] font-semibold text-gray-400 mb-1 uppercase tracking-wider">
                Cloud LLM Inference Provider
              </label>
              <select
                value={selectedProvider}
                onChange={(e) => handleProviderChange(e.target.value)}
                className="w-full bg-[#1e1e1e] border border-gray-800 focus:border-emerald-500 rounded-xl px-3 py-2 text-xs text-gray-100 outline-none"
              >
                <option value="groq">Groq (Ultra-Fast Cloud Inference - Recommended)</option>
                <option value="openai">OpenAI (GPT-4o, GPT-4o-mini)</option>
                <option value="openrouter">OpenRouter (Free & Multi-Provider Hub)</option>
                <option value="gemini">Google Gemini (Gemini 1.5/2.0 Flash)</option>
                <option value="together">Together AI</option>
              </select>
            </div>

            {/* Model Selection */}
            <div>
              <label className="block text-[11px] font-semibold text-gray-400 mb-1 uppercase tracking-wider">
                Model Name
              </label>
              {PROVIDER_PRESETS[selectedProvider]?.models ? (
                <div className="flex gap-2">
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="flex-1 bg-[#1e1e1e] border border-gray-800 focus:border-emerald-500 rounded-xl px-3 py-2 text-xs text-gray-100 outline-none font-mono"
                  >
                    {PROVIDER_PRESETS[selectedProvider].models.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                  <input
                    type="text"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    placeholder="or custom model identifier"
                    className="w-1/3 bg-[#1e1e1e] border border-gray-800 focus:border-emerald-500 rounded-xl px-2.5 py-2 text-xs text-gray-300 font-mono outline-none"
                  />
                </div>
              ) : (
                <input
                  type="text"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full bg-[#1e1e1e] border border-gray-800 focus:border-emerald-500 rounded-xl px-3 py-2 text-xs text-gray-100 font-mono outline-none"
                />
              )}
            </div>

            {/* API Endpoint Base URL */}
            <div>
              <label className="block text-[11px] font-semibold text-gray-400 mb-1 uppercase tracking-wider">
                Hosted Endpoint URL
              </label>
              <input
                type="text"
                value={customBaseUrl}
                onChange={(e) => setCustomBaseUrl(e.target.value)}
                placeholder="https://api.groq.com/openai/v1"
                className="w-full bg-[#1e1e1e] border border-gray-800 focus:border-emerald-500 rounded-xl px-3 py-2 text-xs text-gray-300 font-mono outline-none"
              />
            </div>

            {/* Custom API Key Input */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="block text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
                  Update API Key
                </label>
                {settings?.masked_api_key && settings.masked_api_key !== "No key set" ? (
                  <span className="text-[10px] text-emerald-400 font-mono">
                    Active: {settings.masked_api_key}
                  </span>
                ) : (
                  <span className="text-[10px] text-amber-400 font-medium flex items-center gap-1">
                    ⚠️ No API Key Configured
                  </span>
                )}
              </div>
              <input
                type="password"
                value={customApiKey}
                onChange={(e) => setCustomApiKey(e.target.value)}
                placeholder={
                  PROVIDER_PRESETS[selectedProvider]?.keyPlaceholder || "Enter API Key to update"
                }
                className="w-full bg-[#1e1e1e] border border-gray-800 focus:border-emerald-500 rounded-xl px-3 py-2 text-xs text-gray-100 font-mono outline-none"
              />
              <div className="flex items-center justify-between text-[10px] text-gray-500 mt-1.5">
                <span>
                  {selectedProvider === "groq" ? (
                    <>
                      Need a free key? Get one in 30s at{" "}
                      <a
                        href="https://console.groq.com/keys"
                        target="_blank"
                        rel="noreferrer"
                        className="text-emerald-400 hover:underline"
                      >
                        console.groq.com/keys
                      </a>
                    </>
                  ) : (
                    "Leave blank if using the key configured in backend .env"
                  )}
                </span>
              </div>
            </div>

            {/* Sliders: Temperature, Top-P, Max Tokens */}
            <div className="grid grid-cols-3 gap-2 pt-1">
              <div className="p-2.5 rounded-xl bg-[#1a1a1a] border border-gray-800">
                <div className="flex items-center justify-between text-[11px] text-gray-400 mb-1">
                  <span>Temperature</span>
                  <span className="font-mono text-emerald-400">{temperature}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1.5"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full accent-emerald-500 h-1 bg-gray-700 rounded-lg cursor-pointer"
                />
              </div>

              <div className="p-2.5 rounded-xl bg-[#1a1a1a] border border-gray-800">
                <div className="flex items-center justify-between text-[11px] text-gray-400 mb-1">
                  <span>Top-P</span>
                  <span className="font-mono text-emerald-400">{topP}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={topP}
                  onChange={(e) => setTopP(parseFloat(e.target.value))}
                  className="w-full accent-emerald-500 h-1 bg-gray-700 rounded-lg cursor-pointer"
                />
              </div>

              <div className="p-2.5 rounded-xl bg-[#1a1a1a] border border-gray-800">
                <div className="flex items-center justify-between text-[11px] text-gray-400 mb-1">
                  <span>Max Tokens</span>
                  <span className="font-mono text-emerald-400">{maxTokens}</span>
                </div>
                <input
                  type="range"
                  min="512"
                  max="8192"
                  step="256"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full accent-emerald-500 h-1 bg-gray-700 rounded-lg cursor-pointer"
                />
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <Button
                type="submit"
                size="sm"
                disabled={isSavingLlm}
                className="flex items-center gap-1.5"
              >
                {isSavingLlm ? (
                  <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                ) : (
                  <Save size={13} />
                )}
                <span>Save Hosted LLM Settings</span>
              </Button>
            </div>
          </form>
        )}

        {/* Footer */}
        <div className="flex justify-end pt-3 border-t border-gray-800">
          <Button onClick={onClose} size="sm">
            Done
          </Button>
        </div>
      </div>
    </Modal>
  );
};

