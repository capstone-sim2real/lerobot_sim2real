import * as React from "react"
import { cn } from "@/lib/utils"

// Keep a native select: the agent UI has no client-side React hydration.
function NativeSelect({ className, ...props }: React.ComponentProps<"select">) {
  return <select data-slot="select-trigger" className={cn(
    "h-9 min-w-0 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50",
    className
  )} {...props} />
}
export { NativeSelect }
