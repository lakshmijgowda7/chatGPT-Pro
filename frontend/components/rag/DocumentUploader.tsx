import React, { useState } from "react";
import { X, UploadCloud, CheckCircle2, AlertCircle, FileText } from "lucide-react";
import { API_BASE_URL } from "../../lib/constants";
import { Button } from "../ui/Button";

interface DocumentUploaderProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess: () => void;
}

export const DocumentUploader: React.FC<DocumentUploaderProps> = ({
  isOpen,
  onClose,
  onUploadSuccess,
}) => {
  const [isUploading, setIsUploading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);

  if (!isOpen) return null;

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setStatusMsg(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE_URL}/documents/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");

      setStatusMsg({ type: "success", text: data.message });
      onUploadSuccess();
    } catch (err: any) {
      setStatusMsg({ type: "error", text: err.message || "Upload failed" });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4">
      <div className="bg-[#171717] border border-gray-800 rounded-2xl w-full max-w-md p-6 shadow-2xl space-y-5">
        <div className="flex items-center justify-between border-b border-gray-800 pb-4">
          <h2 className="text-lg font-semibold text-gray-100 flex items-center gap-2">
            <FileText size={18} className="text-[#10a37f]" /> Upload Document (RAG)
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white p-1 rounded-lg">
            <X size={18} />
          </button>
        </div>

        <div className="border-2 border-dashed border-gray-700 hover:border-gray-500 rounded-xl p-6 flex flex-col items-center justify-center gap-3 transition-colors bg-[#212121]/30">
          <UploadCloud size={36} className="text-gray-400" />
          <div className="text-center text-xs text-gray-400">
            <label className="font-semibold text-[#10a37f] hover:underline cursor-pointer">
              <span>Click to choose a file</span>
              <input
                type="file"
                accept=".pdf,.docx,.txt,.md,.csv,.json"
                onChange={handleFileChange}
                disabled={isUploading}
                className="hidden"
              />
            </label>
            <p className="mt-1">Supports PDF, DOCX, TXT, MD, CSV, JSON (Max 25MB)</p>
          </div>
        </div>

        {isUploading && (
          <div className="text-center text-xs text-gray-400 animate-pulse">
            Processing and generating vector embeddings...
          </div>
        )}

        {statusMsg && (
          <div
            className={`p-3 rounded-lg text-xs flex items-center gap-2 ${
              statusMsg.type === "success"
                ? "bg-green-950/40 text-green-300 border border-green-800/40"
                : "bg-red-950/40 text-red-300 border border-red-800/40"
            }`}
          >
            {statusMsg.type === "success" ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
            <span>{statusMsg.text}</span>
          </div>
        )}

        <div className="flex justify-end pt-2">
          <Button onClick={onClose} size="sm">
            Done
          </Button>
        </div>
      </div>
    </div>
  );
};
