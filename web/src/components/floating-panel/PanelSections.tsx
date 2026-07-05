import { IoEyeOutline } from "react-icons/io5";
import { useTranslation } from "react-i18next";
import type { DisplayState, RoiRect, SimulationSettings, SimulationStrategy } from "../../types/simulation";
import { IconButton } from "../IconButton";
import { SettingsSection } from "./SettingsSection";
import { SliderControl } from "./SliderControl";

type PanelSectionsProps = {
  imageHeight: number;
  currentRoi: RoiRect | null;
  strategy: SimulationStrategy;
  display: DisplayState;
  settings: SimulationSettings;
  setStrategy: (strategy: SimulationStrategy) => void;
  updateSetting: <K extends keyof SimulationSettings>(key: K, value: SimulationSettings[K]) => void;
  toggleDisplay: (key: keyof DisplayState) => void;
};

function formatPixels(value: number, fallback: string) {
  return value > 0 ? `${value.toFixed(1)} px` : fallback;
}

export function PanelSections({
  imageHeight,
  currentRoi,
  strategy,
  display,
  settings,
  setStrategy,
  updateSetting,
  toggleDisplay,
}: PanelSectionsProps) {
  const { t } = useTranslation();

  const defaultRoiHalfSizePx = imageHeight > 0 ? imageHeight * settings.defaultRoiHalfSizeRatio : 0;
  const defaultRoiSizePx = defaultRoiHalfSizePx * 2;
  const clearRadiusPx = imageHeight > 0 ? imageHeight * settings.clearRadiusRatio : 0;
  const historySigmaPx = imageHeight > 0 ? imageHeight * settings.historySigmaRatio : 0;
  const distanceSigmaPx = imageHeight > 0 ? imageHeight * settings.distanceSigmaRatio : 0;
  const referenceRoiSizePx = currentRoi?.size ?? defaultRoiSizePx;
  const nmsRadiusModelPx =
    imageHeight > 0 && referenceRoiSizePx > 0
      ? Math.max(1, Math.round((imageHeight * settings.nmsRadiusRatio * 512) / referenceRoiSizePx))
      : 0;
  const imageHeightLabel = imageHeight > 0 ? `${imageHeight} px` : t("panel.unknown");
  const roiSizeLabel = referenceRoiSizePx > 0 ? `${referenceRoiSizePx.toFixed(1)} px` : t("panel.unknown");
  const roiSourceLabel = currentRoi ? t("panel.roiSource.current") : t("panel.roiSource.default");
  const strategyOptions: Array<{ id: SimulationStrategy; label: string }> = [
    { id: "saliency_only", label: t("panel.buttons.saliencyOnlyStrategy") },
    { id: "panel_guided", label: t("panel.buttons.panelGuidedStrategy") },
  ];

  return (
    <div className="settings-sections">
      <SettingsSection
        title={t("panel.sections.panels")}
        toggle={
          <IconButton active={display.showPanelBoxes} label={t("panel.buttons.togglePanels")} onClick={() => toggleDisplay("showPanelBoxes")}>
            <IoEyeOutline />
          </IconButton>
        }
      />

      <SettingsSection
        title={t("panel.sections.preprocess")}
        toggle={
          <IconButton active={display.showPreprocess} label={t("panel.buttons.togglePreprocess")} onClick={() => toggleDisplay("showPreprocess")}>
            <IoEyeOutline />
          </IconButton>
        }
      >
        <SliderControl
          label={t("panel.sliders.blur.label")}
          title={t("panel.sliders.blur.tooltip", { value: settings.maxBlurStrength.toFixed(2) })}
          min="0"
          max="32"
          step="0.25"
          value={settings.maxBlurStrength}
          displayValue={settings.maxBlurStrength.toFixed(2)}
          onChange={(value) => updateSetting("maxBlurStrength", value)}
        />
        <SliderControl
          label={t("panel.sliders.fovea.label")}
          title={t("panel.sliders.fovea.tooltip", {
            imageHeight: imageHeightLabel,
            ratio: settings.clearRadiusRatio.toFixed(3),
            pixels: formatPixels(clearRadiusPx, t("panel.unknown")),
          })}
          min="0.01"
          max="0.2"
          step="0.005"
          value={settings.clearRadiusRatio}
          displayValue={settings.clearRadiusRatio.toFixed(3)}
          onChange={(value) => updateSetting("clearRadiusRatio", value)}
        />
      </SettingsSection>

      <SettingsSection
        title={t("panel.sections.saliency")}
        toggle={
          <IconButton active={display.showHeatmap} label={t("panel.buttons.toggleSaliency")} onClick={() => toggleDisplay("showHeatmap")}>
            <IoEyeOutline />
          </IconButton>
        }
      >
        <SliderControl
          label={t("panel.sliders.box.label")}
          title={t("panel.sliders.box.tooltip", {
            imageHeight: imageHeightLabel,
            ratio: settings.defaultRoiHalfSizeRatio.toFixed(3),
            halfSize: formatPixels(defaultRoiHalfSizePx, t("panel.unknown")),
            fullSize: formatPixels(defaultRoiSizePx, t("panel.unknown")),
          })}
          min="0.05"
          max="0.45"
          step="0.005"
          value={settings.defaultRoiHalfSizeRatio}
          displayValue={settings.defaultRoiHalfSizeRatio.toFixed(3)}
          onChange={(value) => updateSetting("defaultRoiHalfSizeRatio", value)}
        />
      </SettingsSection>

      <SettingsSection
        title={t("panel.sections.history")}
        toggle={
          <IconButton active={display.showHistoryHeatmap} label={t("panel.buttons.toggleHistory")} onClick={() => toggleDisplay("showHistoryHeatmap")}>
            <IoEyeOutline />
          </IconButton>
        }
      >
        <SliderControl
          label={t("panel.sliders.sigma.label")}
          title={t("panel.sliders.sigma.tooltip", {
            imageHeight: imageHeightLabel,
            ratio: settings.historySigmaRatio.toFixed(3),
            pixels: formatPixels(historySigmaPx, t("panel.unknown")),
          })}
          min="0.01"
          max="0.12"
          step="0.001"
          value={settings.historySigmaRatio}
          displayValue={settings.historySigmaRatio.toFixed(3)}
          onChange={(value) => updateSetting("historySigmaRatio", value)}
        />
        <SliderControl
          label={t("panel.sliders.hist.label")}
          title={t("panel.sliders.hist.tooltip", { value: settings.historyAlpha.toFixed(1) })}
          min="0"
          max="8"
          step="0.1"
          value={settings.historyAlpha}
          displayValue={settings.historyAlpha.toFixed(1)}
          onChange={(value) => updateSetting("historyAlpha", value)}
        />
        <SliderControl
          label={t("panel.sliders.decay.label")}
          title={t("panel.sliders.decay.tooltip", { value: settings.historyDecay.toFixed(3) })}
          min="0.7"
          max="0.995"
          step="0.005"
          value={settings.historyDecay}
          displayValue={settings.historyDecay.toFixed(3)}
          onChange={(value) => updateSetting("historyDecay", value)}
        />
      </SettingsSection>

      <SettingsSection
        title={t("panel.sections.selector")}
        toggle={
          <IconButton active={display.showSelectorOverlay} label={t("panel.buttons.toggleSelector")} onClick={() => toggleDisplay("showSelectorOverlay")}>
            <IoEyeOutline />
          </IconButton>
        }
      >
        <div className="text-mode-group" role="group" aria-label={t("panel.sections.selector")}>
          {strategyOptions.map((option) => (
            <button
              key={option.id}
              type="button"
              className={`text-mode-button${strategy === option.id ? " is-active" : ""}`}
              onClick={() => setStrategy(option.id)}
            >
              {option.label}
            </button>
          ))}
        </div>
        <SliderControl
          label={t("panel.sliders.dist.label")}
          title={t("panel.sliders.dist.tooltip", {
            imageHeight: imageHeightLabel,
            ratio: settings.distanceSigmaRatio.toFixed(3),
            pixels: formatPixels(distanceSigmaPx, t("panel.unknown")),
          })}
          min="0.05"
          max="0.4"
          step="0.005"
          value={settings.distanceSigmaRatio}
          displayValue={settings.distanceSigmaRatio.toFixed(3)}
          onChange={(value) => updateSetting("distanceSigmaRatio", value)}
        />
        <SliderControl
          label={t("panel.sliders.thresh.label")}
          title={t("panel.sliders.thresh.tooltip", { value: settings.thresholdRatio.toFixed(2) })}
          min="0.05"
          max="0.95"
          step="0.01"
          value={settings.thresholdRatio}
          displayValue={settings.thresholdRatio.toFixed(2)}
          onChange={(value) => updateSetting("thresholdRatio", value)}
        />
        <SliderControl
          label={t("panel.sliders.nms.label")}
          title={t("panel.sliders.nms.tooltip", {
            imageHeight: imageHeightLabel,
            ratio: settings.nmsRadiusRatio.toFixed(3),
            roiSource: roiSourceLabel,
            roiSize: roiSizeLabel,
            cells: nmsRadiusModelPx,
          })}
          min="0.001"
          max="0.05"
          step="0.001"
          value={settings.nmsRadiusRatio}
          displayValue={settings.nmsRadiusRatio.toFixed(3)}
          onChange={(value) => updateSetting("nmsRadiusRatio", value)}
        />
        <SliderControl
          label={t("panel.sliders.topK.label")}
          title={t("panel.sliders.topK.tooltip", { value: settings.topK.toFixed(0) })}
          min="1"
          max="32"
          step="1"
          value={settings.topK}
          displayValue={settings.topK.toFixed(0)}
          onChange={(value) => updateSetting("topK", value)}
        />
      </SettingsSection>
    </div>
  );
}
