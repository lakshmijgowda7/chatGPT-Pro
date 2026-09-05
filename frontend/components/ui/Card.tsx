import React from "react";
import { cn } from "../../lib/utils";

export const Card: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  children,
  className,
  ...props
}) => {
  return (
    <div
      className={cn(
        "rounded-xl bg-[#212121] border border-gray-800 p-4 text-gray-200 shadow-md",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};
