type LoadingOverlayProps = {
  visible: boolean;
  label: string;
  errorMessage: string | null;
};

export function LoadingOverlay({ visible, label, errorMessage }: LoadingOverlayProps) {
  if (!visible) {
    return null;
  }

  return (
    <div className="loading-overlay" role="status" aria-live="polite">
      <div className="loading-panel">
        <div className="spinner" />
        <div className="loading-label">{errorMessage ?? label}</div>
      </div>
    </div>
  );
}
