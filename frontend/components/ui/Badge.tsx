import React from "react";
import { cn } from "../../lib/utils";

interface BadgeProps {
  children: React.ReactNode;
  variant?: "default" | "success" | "warning" | "danger" | "brand" | "outline";
  size?: "sm" | "md";
  className?: string;
  icon?: React.ReactNode;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = "default",
  size = "sm",
  className,
  icon,
}) => {
  const variants = {
    default: "bg-gray-800 text-gray-300 border-gray-700/60",
    success: "bg-emerald-950/60 text-emerald-300 border-emerald-800/50",
    warning: "bg-amber-950/60 text-amber-300 border-amber-800/50",
    danger: "bg-rose-950/60 text-rose-300 border-rose-800/50",
    brand: "bg-[#10a37f]/15 text-[#10a37f] border-[#10a37f]/30",
    outline: "bg-transparent text-gray-400 border-gray-700",
  };

  const sizes = {
    sm: "px-2 py-0.5 text-[11px]",
    md: "px-2.5 py-1 text-xs",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 font-medium rounded-full border transition-colors",
        variants[variant],
        sizes[size],
        className
      )}
    >
      {icon && <span className="shrink-0">{icon}</span>}
      {children}
    </span>
  );
};
