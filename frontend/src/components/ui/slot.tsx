/** Slot component for composition */
import * as React from "react"

interface SlotProps extends React.HTMLAttributes<HTMLElement> {
  children?: React.ReactNode
}

const Slot = React.forwardRef<HTMLElement, SlotProps>(
  ({ children, ...props }, ref) => {
    if (React.isValidElement(children)) {
      return React.cloneElement(children, {
        ...props,
        ...children.props,
        ref: ref
          ? (composedRef: HTMLElement) => {
              // Handle both the passed ref and child's ref
              if (typeof ref === "function") {
                ref(composedRef)
              } else if (ref) {
                ;(ref as React.MutableRefObject<HTMLElement>).current = composedRef
              }
              // Call child's ref if it exists
              const childRef = (children as React.ReactElement & { ref?: React.Ref<HTMLElement> }).ref
              if (typeof childRef === "function") {
                childRef(composedRef)
              } else if (childRef && typeof childRef === "object") {
                ;(childRef as React.MutableRefObject<HTMLElement>).current = composedRef
              }
            }
          : undefined,
      })
    }

    if (React.Children.count(children) > 1) {
      React.Children.only(null)
    }

    return null
  }
)

Slot.displayName = "Slot"

export { Slot }
