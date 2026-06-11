import { SimulationRuntimeProvider } from "./SimulationRuntime";
import { FloatingPanel } from "../components/FloatingPanel";
import { LoadingOverlay } from "../components/LoadingOverlay";
import { SimulationStage } from "../components/SimulationStage";

export function App() {
  return (
    <SimulationRuntimeProvider>
      <SimulationStage />
      <FloatingPanel />
      <LoadingOverlay />
    </SimulationRuntimeProvider>
  );
}
