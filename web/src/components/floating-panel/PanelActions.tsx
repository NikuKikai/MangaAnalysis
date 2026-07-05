import { useEffect, useRef, useState } from "react";
import { IoLanguageOutline, IoSaveOutline } from "react-icons/io5";
import { TbInfoCircle } from "react-icons/tb";
import { useTranslation } from "react-i18next";
import type { AppLanguage } from "../../i18n/resources";
import { IconButton } from "../IconButton";

type PanelActionsProps = {
  onOpenInfo: () => void;
  onSave: () => void;
};

export function PanelActions({ onOpenInfo, onSave }: PanelActionsProps) {
  const { t, i18n } = useTranslation();
  const languageMenuRef = useRef<HTMLDivElement>(null);
  const [languageMenuOpen, setLanguageMenuOpen] = useState(false);

  useEffect(() => {
    if (!languageMenuOpen) {
      return;
    }

    const handlePointerDown = (event: PointerEvent) => {
      if (languageMenuRef.current && !languageMenuRef.current.contains(event.target as Node)) {
        setLanguageMenuOpen(false);
      }
    };

    window.addEventListener("pointerdown", handlePointerDown);
    return () => window.removeEventListener("pointerdown", handlePointerDown);
  }, [languageMenuOpen]);

  const handleLanguageChange = (language: AppLanguage) => {
    setLanguageMenuOpen(false);
    void i18n.changeLanguage(language);
  };

  return (
    <div className="panel-actions-row">
      <IconButton label={t("panel.buttons.information")} onClick={onOpenInfo}>
        <TbInfoCircle />
      </IconButton>
      <div className="menu-anchor" ref={languageMenuRef}>
        <IconButton active={languageMenuOpen} label={t("panel.buttons.languageMenu")} onClick={() => setLanguageMenuOpen((value) => !value)}>
          <IoLanguageOutline />
        </IconButton>
        {languageMenuOpen ? (
          <div className="panel-menu">
            <button type="button" className="panel-menu-item" onClick={() => handleLanguageChange("en")}>
              {t("panel.language.en")}
            </button>
            <button type="button" className="panel-menu-item" onClick={() => handleLanguageChange("ja")}>
              {t("panel.language.ja")}
            </button>
          </div>
        ) : null}
      </div>
      <IconButton label={t("panel.buttons.save")} onClick={onSave}>
        <IoSaveOutline />
      </IconButton>
    </div>
  );
}
