import type { ReactNode } from "react";

type SettingsSectionProps = {
  title: string;
  toggle?: ReactNode;
  children?: ReactNode;
};

export function SettingsSection({ title, toggle, children }: SettingsSectionProps) {
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
