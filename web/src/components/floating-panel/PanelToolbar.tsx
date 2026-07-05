import { HiBars3 } from "react-icons/hi2";
import { IoFolderOpenOutline, IoPlay, IoRefresh } from "react-icons/io5";
import { LuMousePointer2 } from "react-icons/lu";
import { PiRectangle } from "react-icons/pi";
import { useTranslation } from "react-i18next";
import { IconButton } from "../IconButton";

type PanelToolbarProps = {
  mode: "click" | "box";
  nextDisabled: boolean;
  panelExpanded: boolean;
  onOpenImage: () => void;
  onSetMode: (mode: "click" | "box") => void;
  onNext: () => void;
  onReset: () => void;
  onToggleExpanded: () => void;
};

export function PanelToolbar({
  mode,
  nextDisabled,
  panelExpanded,
  onOpenImage,
  onSetMode,
  onNext,
  onReset,
  onToggleExpanded,
}: PanelToolbarProps) {
  const { t } = useTranslation();

  return (
    <div className="panel-toolbar">
      <IconButton label={t("panel.buttons.openImage")} onClick={onOpenImage}>
        <IoFolderOpenOutline />
      </IconButton>
      <div className="mode-group" role="group" aria-label={`${t("panel.buttons.clickMode")} / ${t("panel.buttons.boxMode")}`}>
        <IconButton active={mode === "click"} label={t("panel.buttons.clickMode")} onClick={() => onSetMode("click")}>
          <LuMousePointer2 />
        </IconButton>
        <IconButton active={mode === "box"} label={t("panel.buttons.boxMode")} onClick={() => onSetMode("box")}>
          <PiRectangle />
        </IconButton>
      </div>
      <IconButton label={t("panel.buttons.nextStep")} disabled={nextDisabled} onClick={onNext}>
        <IoPlay />
      </IconButton>
      <IconButton label={t("panel.buttons.reset")} onClick={onReset}>
        <IoRefresh />
      </IconButton>
      <div className="toolbar-spacer" />
      <IconButton
        active={panelExpanded}
        label={panelExpanded ? t("panel.buttons.collapse") : t("panel.buttons.expand")}
        onClick={onToggleExpanded}
      >
        <HiBars3 />
      </IconButton>
    </div>
  );
}
