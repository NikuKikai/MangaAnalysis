import { useRef, useState, type ChangeEvent, type ReactNode } from "react";
import { HiBars3 } from "react-icons/hi2";
import { IoEyeOutline, IoFolderOpenOutline, IoPlay, IoRefresh } from "react-icons/io5";
import { FaGithub, FaXTwitter } from "react-icons/fa6";
import { LuMousePointer2 } from "react-icons/lu";
import { PiRectangle } from "react-icons/pi";
import { TbInfoCircle } from "react-icons/tb";
import { useShallow } from "zustand/react/shallow";
import { useSimulationEngineContext } from "../app/SimulationRuntime";
import { IconButton } from "./IconButton";
import { useSimulationStore } from "../store/simulationStore";

type SliderControlProps = {
  label: string;
  title: string;
  min: string;
  max: string;
  step: string;
  value: number;
  displayValue: string;
  onChange: (value: number) => void;
};

function SliderControl({ label, title, min, max, step, value, displayValue, onChange }: SliderControlProps) {
  return (
    <label className="slider-row" title={title}>
      <span className="slider-label">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
      <span className="slider-value">{displayValue}</span>
    </label>
  );
}

type SettingsSectionProps = {
  title: string;
  toggle?: ReactNode;
  children: ReactNode;
};

function SettingsSection({ title, toggle, children }: SettingsSectionProps) {
  return (
    <section className="settings-section">
      <div className="settings-section-header">
        <span className="settings-section-title">{title}</span>
        {toggle ? <div className="settings-section-toggle">{toggle}</div> : null}
      </div>
      <div className="settings-section-body">{children}</div>
    </section>
  );
}

export function InfoDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  if (!open) {
    return null;
  }

  return (
    <div className="dialog-backdrop" onClick={onClose} role="presentation">
      <div
        className="dialog-panel"
        role="dialog"
        aria-modal="true"
        aria-label="Information"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="dialog-content">
          <div className="info-card">
            <p className="info-line">Author: NikuKikai</p>
            <p className="info-line info-contact">Contact: nikukikai@gmail.com</p>
            <div className="info-links" aria-label="Social links">
              <a
                className="info-link"
                href="https://github.com/nikukikai"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="GitHub"
                title="GitHub"
              >
                <FaGithub />
              </a>
              <a
                className="info-link"
                href="https://x.com/nikukikai"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="X"
                title="X"
              >
                <FaXTwitter />
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function FloatingPanel() {
  const { handleNextStep } = useSimulationEngineContext();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [panelExpanded, setPanelExpanded] = useState(true);
  const [infoOpen, setInfoOpen] = useState(false);
  const {
    image,
    currentRoi,
    mode,
    display,
    settings,
    pendingNextFixation,
    loadImageFile,
    setMode,
    updateSetting,
    toggleDisplay,
    clearSimulation,
  } =
    useSimulationStore(
      useShallow((state) => ({
        image: state.image,
        currentRoi: state.currentRoi,
        mode: state.mode,
        display: state.display,
        settings: state.settings,
        pendingNextFixation: state.pendingNextFixation,
        loadImageFile: state.loadImageFile,
        setMode: state.setMode,
        updateSetting: state.updateSetting,
        toggleDisplay: state.toggleDisplay,
        clearSimulation: state.clearSimulation,
      })),
    );
  const nextDisabled = !pendingNextFixation;
  const openImageDialog = () => fileInputRef.current?.click();
  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    void loadImageFile(event.target.files?.[0] ?? null);
    event.target.value = "";
  };
  const imageHeight = image?.height ?? 0;
  const clickRoiHalfSizePx = imageHeight > 0 ? imageHeight * settings.clickRoiHalfSizeRatio : 0;
  const clickRoiSizePx = clickRoiHalfSizePx * 2;
  const clearRadiusPx = imageHeight > 0 ? imageHeight * settings.clearRadiusRatio : 0;
  const historySigmaPx = imageHeight > 0 ? imageHeight * settings.historySigmaRatio : 0;
  const distanceSigmaPx = imageHeight > 0 ? imageHeight * settings.distanceSigmaRatio : 0;
  const referenceRoiSizePx = currentRoi?.size ?? clickRoiSizePx;
  const nmsRadiusModelPx =
    imageHeight > 0 && referenceRoiSizePx > 0
      ? Math.max(1, Math.round((imageHeight * settings.nmsRadiusRatio * 512) / referenceRoiSizePx))
      : 0;
  const formatPixels = (value: number) => (value > 0 ? `${value.toFixed(1)} px` : "unknown");
  const imageHeightLabel = imageHeight > 0 ? `${imageHeight} px` : "unknown";
  const roiSizeLabel = referenceRoiSizePx > 0 ? `${referenceRoiSizePx.toFixed(1)} px` : "unknown";
  const roiSourceLabel = currentRoi ? "current ROI size" : "default click ROI size";

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
        <div className="panel-toolbar">
          <IconButton label="Open Image" onClick={openImageDialog}>
            <IoFolderOpenOutline />
          </IconButton>
          <div className="mode-group" role="group" aria-label="Mouse mode">
            <IconButton active={mode === "click"} label="Click Mode" onClick={() => setMode("click")}>
              <LuMousePointer2 />
            </IconButton>
            <IconButton active={mode === "box"} label="Box Mode" onClick={() => setMode("box")}>
              <PiRectangle />
            </IconButton>
          </div>
          <IconButton label="Next Step" disabled={nextDisabled} onClick={() => void handleNextStep()}>
            <IoPlay />
          </IconButton>
          <IconButton label="Reset" onClick={clearSimulation}>
            <IoRefresh />
          </IconButton>
          <div className="toolbar-spacer" />
          <IconButton label="Information" onClick={() => setInfoOpen(true)}>
            <TbInfoCircle />
          </IconButton>
          <IconButton
            active={panelExpanded}
            label={panelExpanded ? "Collapse Panel" : "Expand Panel"}
            onClick={() => setPanelExpanded((value) => !value)}
          >
            <HiBars3 />
          </IconButton>
        </div>

        {panelExpanded ? (
          <div className="settings-sections">
            <SettingsSection
              title="preprocess"
              toggle={
                <IconButton active={display.showPreprocess} label="Toggle Preprocess View" onClick={() => toggleDisplay("showPreprocess")}>
                  <IoEyeOutline />
                </IconButton>
              }
            >
              <SliderControl
                label="Blur"
                title={`Maximum blur radius used outside\nthe clear fovea.\nCurrent value: ${settings.maxBlurStrength.toFixed(2)} px.\nLarger values make peripheral regions blurrier\nand harder to read.\nSmaller values keep more detail.`}
                min="0"
                max="32"
                step="0.25"
                value={settings.maxBlurStrength}
                displayValue={settings.maxBlurStrength.toFixed(2)}
                onChange={(value) => updateSetting("maxBlurStrength", value)}
              />
              <SliderControl
                label="Fovea"
                title={`Radius of the sharp center area\naround the fixation point.\nFormula: image height (${imageHeightLabel})\nx ratio (${settings.clearRadiusRatio.toFixed(3)})\n= ${formatPixels(clearRadiusPx)}.\nLarger values keep a wider area unblurred.\nSmaller values make blur start closer\nto the fixation point.`}
                min="0.01"
                max="0.2"
                step="0.005"
                value={settings.clearRadiusRatio}
                displayValue={settings.clearRadiusRatio.toFixed(3)}
                onChange={(value) => updateSetting("clearRadiusRatio", value)}
              />
            </SettingsSection>

            <SettingsSection
              title="saliency"
              toggle={
                <IconButton active={display.showHeatmap} label="Toggle Saliency View" onClick={() => toggleDisplay("showHeatmap")}>
                  <IoEyeOutline />
                </IconButton>
              }
            >
              <SliderControl
                label="Box"
                title={`Half-size of the square ROI used in click mode\nand as the default size for Next Step.\nFormula: image height (${imageHeightLabel})\nx ratio (${settings.clickRoiHalfSizeRatio.toFixed(3)})\n= half-size ${formatPixels(clickRoiHalfSizePx)}.\nFull ROI size is ${formatPixels(clickRoiSizePx)}.\nLarger values inspect a wider area\nwith lower local detail.\nSmaller values focus on a tighter\nlocal neighborhood.`}
                min="0.05"
                max="0.45"
                step="0.005"
                value={settings.clickRoiHalfSizeRatio}
                displayValue={settings.clickRoiHalfSizeRatio.toFixed(3)}
                onChange={(value) => updateSetting("clickRoiHalfSizeRatio", value)}
              />
            </SettingsSection>

            <SettingsSection
              title="history"
              toggle={
                <IconButton active={display.showHistoryHeatmap} label="Toggle History View" onClick={() => toggleDisplay("showHistoryHeatmap")}>
                  <IoEyeOutline />
                </IconButton>
              }
            >
              <SliderControl
                label="Sigma"
                title={`Spread of each fixation deposit written\ninto the history map.\nFormula: image height (${imageHeightLabel})\nx ratio (${settings.historySigmaRatio.toFixed(3)})\n= sigma ${formatPixels(historySigmaPx)}.\nLarger values spread inhibition across\na broader area.\nSmaller values keep inhibition concentrated\nnear the fixation point.`}
                min="0.01"
                max="0.12"
                step="0.001"
                value={settings.historySigmaRatio}
                displayValue={settings.historySigmaRatio.toFixed(3)}
                onChange={(value) => updateSetting("historySigmaRatio", value)}
              />
              <SliderControl
                label="Hist"
                title={`Strength of history-based suppression\nin candidate scoring.\nCurrent value: ${settings.historyAlpha.toFixed(1)}.\nThe score multiplier is\nexp(-alpha x historyValue).\nLarger values penalize revisits more aggressively.\nSmaller values make history matter less.`}
                min="0"
                max="8"
                step="0.1"
                value={settings.historyAlpha}
                displayValue={settings.historyAlpha.toFixed(1)}
                onChange={(value) => updateSetting("historyAlpha", value)}
              />
              <SliderControl
                label="Decay"
                title={`How quickly old history fades\nwhen a new fixation is added.\nCurrent value: ${settings.historyDecay.toFixed(3)}.\nNew history is accumulated as\noldValue x decay + gaussian.\nLarger values preserve history longer.\nSmaller values erase old inhibition faster.`}
                min="0.7"
                max="0.995"
                step="0.005"
                value={settings.historyDecay}
                displayValue={settings.historyDecay.toFixed(3)}
                onChange={(value) => updateSetting("historyDecay", value)}
              />
            </SettingsSection>

            <SettingsSection title="selector">
              <SliderControl
                label="Dist"
                title={`Spread of the distance preference\naround the current fixation.\nFormula: image height (${imageHeightLabel})\nx ratio (${settings.distanceSigmaRatio.toFixed(3)})\n= sigma ${formatPixels(distanceSigmaPx)}.\nLarger values weaken the preference\nfor nearby candidates.\nSmaller values bias the next fixation\nmore strongly toward nearby locations.`}
                min="0.05"
                max="0.4"
                step="0.005"
                value={settings.distanceSigmaRatio}
                displayValue={settings.distanceSigmaRatio.toFixed(3)}
                onChange={(value) => updateSetting("distanceSigmaRatio", value)}
              />
              <SliderControl
                label="Thresh"
                title={`Post-filter threshold applied\nto final candidate scores.\nCurrent value: ${settings.thresholdRatio.toFixed(2)}.\nCandidates with finalScore below this value\nare discarded.\nLarger values keep only stronger peaks.\nSmaller values allow weaker candidates\nto survive.`}
                min="0.05"
                max="0.95"
                step="0.01"
                value={settings.thresholdRatio}
                displayValue={settings.thresholdRatio.toFixed(2)}
                onChange={(value) => updateSetting("thresholdRatio", value)}
              />
              <SliderControl
                label="Nms"
                title={`Non-maximum suppression radius\nin model-space cells.\nCurrent estimate:\nround(image height (${imageHeightLabel})\nx ratio (${settings.nmsRadiusRatio.toFixed(3)})\nx model size 512\n/ ROI size (${roiSourceLabel}: ${roiSizeLabel}))\n= ${nmsRadiusModelPx} cells.\nLarger values merge nearby peaks\nmore aggressively.\nSmaller values keep more local maxima.`}
                min="0.001"
                max="0.05"
                step="0.001"
                value={settings.nmsRadiusRatio}
                displayValue={settings.nmsRadiusRatio.toFixed(3)}
                onChange={(value) => updateSetting("nmsRadiusRatio", value)}
              />
              <SliderControl
                label="TopK"
                title={`Maximum number of candidates kept\nafter GPU ranking and sorting.\nCurrent value: ${settings.topK.toFixed(0)}.\nLarger values preserve more alternatives\nfor inspection.\nSmaller values keep only the strongest\nfew candidates.`}
                min="1"
                max="32"
                step="1"
                value={settings.topK}
                displayValue={settings.topK.toFixed(0)}
                onChange={(value) => updateSetting("topK", value)}
              />
            </SettingsSection>
          </div>
        ) : null}
      </div>
      <InfoDialog open={infoOpen} onClose={() => setInfoOpen(false)} />
    </>
  );
}
