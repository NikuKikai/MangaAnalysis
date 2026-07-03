import { useRef, useState, type ChangeEvent, type ReactNode } from "react";
import { HiBars3 } from "react-icons/hi2";
import { IoEyeOutline, IoFolderOpenOutline, IoPlay, IoRefresh } from "react-icons/io5";
import { FaGithub, FaXTwitter } from "react-icons/fa6";
import { LuMousePointer2 } from "react-icons/lu";
import { PiRectangle } from "react-icons/pi";
import { TbInfoCircle } from "react-icons/tb";
import { useTranslation } from "react-i18next";
import { useShallow } from "zustand/react/shallow";
import { useSimulationEngineContext } from "../app/SimulationRuntime";
import { IconButton } from "./IconButton";
import type { AppLanguage } from "../i18n/resources";
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
  const { t } = useTranslation();
  if (!open) {
    return null;
  }

  return (
    <div className="dialog-backdrop" onClick={onClose} role="presentation">
      <div
        className="dialog-panel"
        role="dialog"
        aria-modal="true"
        aria-label={t("panel.info.dialogLabel")}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="dialog-content">
          <div className="info-card">
            <p className="info-line">{t("panel.info.author")}</p>
            <p className="info-line info-contact">{t("panel.info.contact")}</p>
            <div className="info-links" aria-label={t("panel.info.socialLinks")}>
              <a
                className="info-link"
                href="https://github.com/nikukikai"
                target="_blank"
                rel="noopener noreferrer"
                aria-label={t("panel.info.github")}
                title={t("panel.info.github")}
              >
                <FaGithub />
              </a>
              <a
                className="info-link"
                href="https://x.com/nikukikai"
                target="_blank"
                rel="noopener noreferrer"
                aria-label={t("panel.info.x")}
                title={t("panel.info.x")}
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
  const { t, i18n } = useTranslation();
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
  const formatPixels = (value: number) => (value > 0 ? `${value.toFixed(1)} px` : t("panel.unknown"));
  const imageHeightLabel = imageHeight > 0 ? `${imageHeight} px` : t("panel.unknown");
  const roiSizeLabel = referenceRoiSizePx > 0 ? `${referenceRoiSizePx.toFixed(1)} px` : t("panel.unknown");
  const roiSourceLabel = currentRoi ? t("panel.roiSource.current") : t("panel.roiSource.default");
  const handleLanguageChange = (event: ChangeEvent<HTMLSelectElement>) => {
    void i18n.changeLanguage(event.target.value as AppLanguage);
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
        <div className="panel-toolbar">
          <IconButton label={t("panel.buttons.openImage")} onClick={openImageDialog}>
            <IoFolderOpenOutline />
          </IconButton>
          <div className="mode-group" role="group" aria-label={`${t("panel.buttons.clickMode")} / ${t("panel.buttons.boxMode")}`}>
            <IconButton active={mode === "click"} label={t("panel.buttons.clickMode")} onClick={() => setMode("click")}>
              <LuMousePointer2 />
            </IconButton>
            <IconButton active={mode === "box"} label={t("panel.buttons.boxMode")} onClick={() => setMode("box")}>
              <PiRectangle />
            </IconButton>
          </div>
          <IconButton label={t("panel.buttons.nextStep")} disabled={nextDisabled} onClick={() => void handleNextStep()}>
            <IoPlay />
          </IconButton>
          <IconButton label={t("panel.buttons.reset")} onClick={clearSimulation}>
            <IoRefresh />
          </IconButton>
          <div className="toolbar-spacer" />
          <IconButton label={t("panel.buttons.information")} onClick={() => setInfoOpen(true)}>
            <TbInfoCircle />
          </IconButton>
          <IconButton
            active={panelExpanded}
            label={panelExpanded ? t("panel.buttons.collapse") : t("panel.buttons.expand")}
            onClick={() => setPanelExpanded((value) => !value)}
          >
            <HiBars3 />
          </IconButton>
        </div>
        {panelExpanded ? (
          <div className="settings-sections">
            <div className="panel-language-row">
              <label className="panel-language-label" htmlFor="language-select">
                {t("panel.language.label")}
              </label>
              <select
                id="language-select"
                className="panel-language-select"
                value={(i18n.resolvedLanguage ?? i18n.language) as AppLanguage}
                onChange={handleLanguageChange}
              >
                <option value="en">{t("panel.language.en")}</option>
                <option value="ja">{t("panel.language.ja")}</option>
              </select>
            </div>
            <SettingsSection
              title={t("panel.sections.panels")}
              toggle={
                <IconButton active={display.showPanelBoxes} label={t("panel.buttons.togglePanels")} onClick={() => toggleDisplay("showPanelBoxes")}>
                  <IoEyeOutline />
                </IconButton>
              }
            >
              <p className="settings-note">{t("panel.sections.panelBoxesNote")}</p>
            </SettingsSection>

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
                  pixels: formatPixels(clearRadiusPx),
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
                  halfSize: formatPixels(defaultRoiHalfSizePx),
                  fullSize: formatPixels(defaultRoiSizePx),
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
                  pixels: formatPixels(historySigmaPx),
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

            <SettingsSection title={t("panel.sections.selector")}>
              <SliderControl
                label={t("panel.sliders.dist.label")}
                title={t("panel.sliders.dist.tooltip", {
                  imageHeight: imageHeightLabel,
                  ratio: settings.distanceSigmaRatio.toFixed(3),
                  pixels: formatPixels(distanceSigmaPx),
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
        ) : null}
      </div>
      <InfoDialog open={infoOpen} onClose={() => setInfoOpen(false)} />
    </>
  );
}
