import React from "react";
import { cn } from "../../lib/utils";

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  icon?: React.ReactNode;
  rightElement?: React.ReactNode;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type = "text", icon, rightElement, ...props }, ref) => {
    return (
      <div className="relative flex items-center w-full">
        {icon && (
          <div className="absolute left-3 flex items-center pointer-events-none text-gray-400">
            {icon}
          </div>
        )}
        <input
          type={type}
          ref={ref}
          className={cn(
            "w-full rounded-xl bg-[#212121] border border-gray-700/80 px-3 py-2 text-sm text-gray-100 placeholder-gray-500",
            "focus:outline-none focus:border-[#10a37f] focus:ring-1 focus:ring-[#10a37f] transition-all",
            "disabled:cursor-not-allowed disabled:opacity-50",
            icon ? "pl-9" : "",
            rightElement ? "pr-9" : "",
            className
          )}
          {...props}
        />
        {rightElement && (
          <div className="absolute right-2.5 flex items-center">
            {rightElement}
          </div>
        )}
      </div>
    );
  }
);

Input.displayName = "Input";
