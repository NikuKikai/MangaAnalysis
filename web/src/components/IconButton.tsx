import type { ReactNode } from "react";

type IconButtonProps = {
  active?: boolean;
  disabled?: boolean;
  label: string;
  onClick?: () => void;
  children: ReactNode;
};

export function IconButton({ active = false, disabled = false, label, onClick, children }: IconButtonProps) {
  return (
    <button
      type="button"
      className={`icon-button${active ? " is-active" : ""}`}
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </button>
  );
}
