import * as React from "react"
import { cn } from "@/lib/utils"

// Native input keeps keyboard and label behavior without React hydration.
function NativeCheckbox({ className, ...props }: React.ComponentProps<"input">) {
  return <input type="checkbox" data-slot="checkbox" className={cn(
    "size-4 shrink-0 rounded border border-input accent-primary focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50",
    className
  )} {...props} />
}
export { NativeCheckbox }
