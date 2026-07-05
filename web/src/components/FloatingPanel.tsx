import { useRef, useState, type ChangeEvent } from "react";
import { useShallow } from "zustand/react/shallow";
import { useSimulationEngineContext } from "../app/SimulationRuntime";
import { useSimulationStore } from "../store/simulationStore";
import { PanelActions } from "./floating-panel/PanelActions";
import { InfoDialog } from "./floating-panel/InfoDialog";
import { PanelSections } from "./floating-panel/PanelSections";
import { PanelToolbar } from "./floating-panel/PanelToolbar";
import { exportVisualization } from "./floating-panel/export/exportVisualization";

export function FloatingPanel() {
  const { refs, exportBuffers, overlay, handleNextStep } = useSimulationEngineContext();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [panelExpanded, setPanelExpanded] = useState(true);
  const [infoOpen, setInfoOpen] = useState(false);
  const {
    image,
    currentRoi,
    currentFixation,
    pendingNextFixation,
    candidates,
    panelBoxes,
    panelGuidedAnalysis,
    trajectory,
    mode,
    strategy,
    display,
    settings,
    loadImageFile,
    setMode,
    setStrategy,
    updateSetting,
    toggleDisplay,
    clearSimulation,
  } =
    useSimulationStore(
      useShallow((state) => ({
        image: state.image,
        currentRoi: state.currentRoi,
        currentFixation: state.currentFixation,
        pendingNextFixation: state.pendingNextFixation,
        candidates: state.candidates,
        panelBoxes: state.panelBoxes,
        panelGuidedAnalysis: state.panelGuidedAnalysis,
        trajectory: state.trajectory,
        mode: state.mode,
        strategy: state.strategy,
        display: state.display,
        settings: state.settings,
        loadImageFile: state.loadImageFile,
        setMode: state.setMode,
        setStrategy: state.setStrategy,
        updateSetting: state.updateSetting,
        toggleDisplay: state.toggleDisplay,
        clearSimulation: state.clearSimulation,
      })),
    );

  const openImageDialog = () => fileInputRef.current?.click();

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    void loadImageFile(event.target.files?.[0] ?? null);
    event.target.value = "";
  };

  const handleSaveVisualization = async () => {
    await exportVisualization({
      image,
      refs: {
        historyCanvas: refs.historyCanvasRef.current,
      },
      exportBuffers,
      overlay,
      simulation: {
        currentRoi,
        currentFixation,
        pendingNextFixation,
        trajectory,
        candidates,
        panelBoxes,
        panelGuidedAnalysis,
        strategy,
        display,
      },
    });
  };

  return (
    <>
      <input
        ref={fileInputRef}
        className="visually-hidden"
        type="file"
        accept="image/*"
        onChange={handleFileChange}
      />
      <div className="panel panel-top-right panel-control">
        <PanelToolbar
          mode={mode}
          nextDisabled={!pendingNextFixation}
          panelExpanded={panelExpanded}
          onOpenImage={openImageDialog}
          onSetMode={setMode}
          onNext={() => void handleNextStep()}
          onReset={clearSimulation}
          onToggleExpanded={() => setPanelExpanded((value) => !value)}
        />
        {panelExpanded ? (
          <>
            <PanelActions onOpenInfo={() => setInfoOpen(true)} onSave={() => void handleSaveVisualization()} />
            <PanelSections
              imageHeight={image?.height ?? 0}
              currentRoi={currentRoi}
              strategy={strategy}
              display={display}
              settings={settings}
              setStrategy={setStrategy}
              updateSetting={updateSetting}
              toggleDisplay={toggleDisplay}
            />
          </>
        ) : null}
      </div>
      <InfoDialog open={infoOpen} onClose={() => setInfoOpen(false)} />
    </>
  );
}
