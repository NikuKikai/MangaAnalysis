import { IconButton } from "./IconButton";
import type { DisplayState, MouseMode, SimulationSettings } from "../types/simulation";

type FloatingPanelProps = {
  mode: MouseMode;
  display: DisplayState;
  settings: SimulationSettings;
  onModeChange: (mode: MouseMode) => void;
  onSettingChange: <K extends keyof SimulationSettings>(key: K, value: SimulationSettings[K]) => void;
  onToggleDisplay: (key: keyof DisplayState) => void;
  onNextStep: () => void;
  onReset: () => void;
  onOpenImage: () => void;
  nextDisabled: boolean;
};

function CursorIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M6 3L18 14H13L15.5 21L12.8 22L10.3 15H6V3Z" />
    </svg>
  );
}

function BoxIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 5H19V19H5V5ZM7 7V17H17V7H7Z" />
    </svg>
  );
}

function FolderIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 6H10L12 8H20V18H4V6ZM6 10V16H18V10H6Z" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M8 5L19 12L8 19V5Z" />
    </svg>
  );
}

function HeatIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3C14 5.2 16 7.4 16 11A4 4 0 1 1 8 11C8 9 8.8 7.4 10.1 5.9C10.6 7.1 11.4 8 12.5 8.7C12.7 6.7 12.4 5 12 3ZM12 13.2A1.8 1.8 0 1 0 12 16.8A1.8 1.8 0 0 0 12 13.2Z" />
    </svg>
  );
}

function EyeIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 6C17 6 20.7 10.2 21.7 12C20.7 13.8 17 18 12 18C7 18 3.3 13.8 2.3 12C3.3 10.2 7 6 12 6ZM12 8A4 4 0 1 0 12 16A4 4 0 0 0 12 8Z" />
    </svg>
  );
}

function PathIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M6 18A2 2 0 1 0 6 14A2 2 0 0 0 6 18ZM18 10A2 2 0 1 0 18 6A2 2 0 0 0 18 10ZM12 14A2 2 0 1 0 12 10A2 2 0 0 0 12 14ZM7.4 14.6L10.6 11.4L11.4 12.2L8.2 15.4L7.4 14.6ZM13.4 11.4L16.6 8.2L17.4 9L14.2 12.2L13.4 11.4Z" />
    </svg>
  );
}

function ResetIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 5A7 7 0 1 1 5 12H3A9 9 0 1 0 5.6 5.6L3 8.2V3H8.2L6.9 4.3A8.9 8.9 0 0 1 12 5Z" />
    </svg>
  );
}

export function FloatingPanel(props: FloatingPanelProps) {
  const {
    mode,
    display,
    settings,
    onModeChange,
    onSettingChange,
    onToggleDisplay,
    onNextStep,
    onReset,
    onOpenImage,
    nextDisabled,
  } = props;

  return (
    <>
      <div className="panel panel-top-left">
        <IconButton label="Open Image" onClick={onOpenImage}>
          <FolderIcon />
        </IconButton>
      </div>

      <div className="panel panel-top-right">
        <div className="mode-group" role="group" aria-label="Mouse mode">
          <IconButton active={mode === "click"} label="Click Mode" onClick={() => onModeChange("click")}>
            <CursorIcon />
          </IconButton>
          <IconButton active={mode === "box"} label="Box Mode" onClick={() => onModeChange("box")}>
            <BoxIcon />
          </IconButton>
        </div>
        <IconButton label="Next Step" disabled={nextDisabled} onClick={onNextStep}>
          <PlayIcon />
        </IconButton>
        <IconButton active={display.showPreprocess} label="Preprocess" onClick={() => onToggleDisplay("showPreprocess")}>
          <EyeIcon />
        </IconButton>
        <IconButton active={display.showHeatmap} label="Heatmap" onClick={() => onToggleDisplay("showHeatmap")}>
          <HeatIcon />
        </IconButton>
        <IconButton active={display.showHistory} label="History" onClick={() => onToggleDisplay("showHistory")}>
          <PathIcon />
        </IconButton>
        <IconButton label="Reset" onClick={onReset}>
          <ResetIcon />
        </IconButton>
      </div>

      <div className="panel panel-bottom-right sliders">
        <label className="slider-row" title="Max Blur Strength">
          <span className="slider-label">
            Blur
          </span>
          <input
            type="range"
            min="0"
            max="32"
            step="0.25"
            value={settings.maxBlurStrength}
            onChange={(event) => onSettingChange("maxBlurStrength", Number(event.target.value))}
          />
          <span className="slider-value">{settings.maxBlurStrength.toFixed(2)}</span>
        </label>
        <label className="slider-row" title="Clear Radius Ratio">
          <span className="slider-label">
            Fovea
          </span>
          <input
            type="range"
            min="0.01"
            max="0.2"
            step="0.005"
            value={settings.clearRadiusRatio}
            onChange={(event) => onSettingChange("clearRadiusRatio", Number(event.target.value))}
          />
          <span className="slider-value">{settings.clearRadiusRatio.toFixed(3)}</span>
        </label>
        <label className="slider-row" title="Click ROI Half Size Ratio">
          <span className="slider-label">
            Box
          </span>
          <input
            type="range"
            min="0.05"
            max="0.45"
            step="0.005"
            value={settings.clickRoiHalfSizeRatio}
            onChange={(event) => onSettingChange("clickRoiHalfSizeRatio", Number(event.target.value))}
          />
          <span className="slider-value">{settings.clickRoiHalfSizeRatio.toFixed(3)}</span>
        </label>
      </div>
    </>
  );
}
