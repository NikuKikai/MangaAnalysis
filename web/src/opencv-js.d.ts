type OpenCvApi = {
  onRuntimeInitialized?: (() => void) | null;
  [key: string]: unknown;
};

declare global {
  interface Window {
    cv?: OpenCvApi | Promise<OpenCvApi>;
    Module?: {
      onRuntimeInitialized?: (() => void) | null;
    };
  }
}

export {};
