import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import { resources, type AppLanguage } from "./resources";

const LANGUAGE_STORAGE_KEY = "manga-analysis-language";
const DEFAULT_LANGUAGE: AppLanguage = "en";

function detectLanguage(): AppLanguage {
  const saved = window.localStorage.getItem(LANGUAGE_STORAGE_KEY);
  if (saved === "en" || saved === "ja") {
    return saved;
  }

  const browserLanguage = window.navigator.language.toLowerCase();
  return browserLanguage.startsWith("ja") ? "ja" : DEFAULT_LANGUAGE;
}

void i18n.use(initReactI18next).init({
  resources,
  lng: detectLanguage(),
  fallbackLng: DEFAULT_LANGUAGE,
  interpolation: {
    escapeValue: false,
  },
});

i18n.on("languageChanged", (language) => {
  window.localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
});

export { LANGUAGE_STORAGE_KEY };
export default i18n;
