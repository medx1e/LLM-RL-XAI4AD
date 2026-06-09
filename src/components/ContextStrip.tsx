import { ReactNode } from "react";
import { SidebarTrigger } from "@/components/ui/sidebar";

export function ContextStrip({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children?: ReactNode;
}) {
  return (
    <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-border bg-background/80 px-4 backdrop-blur-md">
      <SidebarTrigger />
      <div className="flex items-baseline gap-3 min-w-0">
        <h1 className="truncate text-base font-semibold tracking-tight">{title}</h1>
        {subtitle && (
          <span className="hidden sm:inline mono text-[11px] text-muted-foreground truncate">
            {subtitle}
          </span>
        )}
      </div>
      <div className="ml-auto flex items-center gap-2">{children}</div>
    </header>
  );
}
