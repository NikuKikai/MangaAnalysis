import { SimulationProvider } from "./SimulationRuntime";
import { FloatingPanel } from "../components/FloatingPanel";
import { LoadingOverlay } from "../components/LoadingOverlay";
import { SimulationStage } from "../components/SimulationStage";

export function App() {
  return (
    <SimulationProvider>
      <SimulationStage />
      <FloatingPanel />
      <LoadingOverlay />
    </SimulationProvider>
  );
}
