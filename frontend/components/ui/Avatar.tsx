import React from "react";
import { cn } from "../../lib/utils";
import { User, Sparkles, Bot } from "lucide-react";

interface AvatarProps {
  role: "user" | "assistant" | "system";
  size?: "sm" | "md" | "lg";
  className?: string;
  isOnline?: boolean;
}

export const Avatar: React.FC<AvatarProps> = ({
  role,
  size = "md",
  className,
  isOnline,
}) => {
  const sizeClasses = {
    sm: "w-6 h-6 text-xs",
    md: "w-8 h-8 text-sm",
    lg: "w-10 h-10 text-base",
  };

  const iconSizes = {
    sm: 12,
    md: 16,
    lg: 20,
  };

  const isUser = role === "user";

  return (
    <div className="relative shrink-0">
      <div
        className={cn(
          "rounded-full flex items-center justify-center font-medium shadow-md transition-transform",
          sizeClasses[size],
          isUser
            ? "bg-gradient-to-br from-indigo-500 to-purple-600 text-white"
            : "bg-gradient-to-br from-emerald-500 to-[#10a37f] text-white shadow-emerald-500/10",
          className
        )}
      >
        {isUser ? (
          <User size={iconSizes[size]} />
        ) : (
          <Sparkles size={iconSizes[size]} />
        )}
      </div>

      {isOnline !== undefined && (
        <span
          className={cn(
            "absolute bottom-0 right-0 rounded-full ring-2 ring-[#171717]",
            size === "sm" ? "w-2 h-2" : "w-2.5 h-2.5",
            isOnline ? "bg-emerald-500" : "bg-gray-500"
          )}
        />
      )}
    </div>
  );
};
