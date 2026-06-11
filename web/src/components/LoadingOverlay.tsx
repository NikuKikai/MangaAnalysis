import { useShallow } from "zustand/react/shallow";
import { useSimulationStore } from "../store/simulationStore";

export function LoadingOverlay() {
  const { loadingPhase, loadingLabel, errorMessage } = useSimulationStore(
    useShallow((state) => ({
      loadingPhase: state.loadingPhase,
      loadingLabel: state.loadingLabel,
      errorMessage: state.errorMessage,
    })),
  );
  const visible = loadingPhase !== "ready";
  if (!visible) {
    return null;
  }

  return (
    <div className="loading-overlay" role="status" aria-live="polite">
      <div className="loading-panel">
        <div className="spinner" />
        <div className="loading-label">{errorMessage ?? loadingLabel}</div>
      </div>
    </div>
  );
}
