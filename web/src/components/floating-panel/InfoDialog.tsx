import { FaGithub, FaXTwitter } from "react-icons/fa6";
import { useTranslation } from "react-i18next";

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
