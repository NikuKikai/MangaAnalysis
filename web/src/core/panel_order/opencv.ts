const OPENCV_CDN_URL = "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.11.0-release.1/dist/opencv.js";

let cvPromise: Promise<OpenCvApi> | null = null;

function resolveLoadedCv(): OpenCvApi | null {
  const loadedCv = window.cv;
  if (!loadedCv || typeof (loadedCv as Promise<OpenCvApi>).then === "function") {
    return null;
  }
  return loadedCv as OpenCvApi;
}

function waitForRuntimeInitialization(): Promise<OpenCvApi> {
  return new Promise((resolve, reject) => {
    const loadedCv = window.cv;
    if (loadedCv && typeof (loadedCv as Promise<OpenCvApi>).then === "function") {
      void (loadedCv as Promise<OpenCvApi>).then((resolvedCv) => {
        window.cv = resolvedCv;
        resolve(resolvedCv);
      }, reject);
      return;
    }

    const readyCv = resolveLoadedCv();
    if (readyCv && typeof readyCv.Mat === "function") {
      resolve(readyCv);
      return;
    }

    const handleReady = () => {
      const resolvedCv = resolveLoadedCv();
      if (!resolvedCv) {
        reject(new Error("OpenCV.js runtime initialized without exposing window.cv."));
        return;
      }
      resolvedCv.onRuntimeInitialized = null;
      resolve(resolvedCv);
    };

    window.Module = {
      ...(window.Module ?? {}),
      onRuntimeInitialized: handleReady,
    };

    if (readyCv) {
      readyCv.onRuntimeInitialized = handleReady;
    }
  });
}

export function loadOpenCv(): Promise<OpenCvApi> {
  if (cvPromise) {
    return cvPromise;
  }

  cvPromise = new Promise((resolve, reject) => {
    const readyCv = resolveLoadedCv();
    if (readyCv && typeof readyCv.Mat === "function") {
      resolve(readyCv);
      return;
    }

    const existingScript = document.querySelector<HTMLScriptElement>('script[data-opencv-js="true"]');
    if (existingScript) {
      void waitForRuntimeInitialization().then(resolve, reject);
      return;
    }

    const script = document.createElement("script");
    script.src = OPENCV_CDN_URL;
    script.async = true;
    script.setAttribute("data-opencv-js", "true");
    script.addEventListener("load", () => {
      void waitForRuntimeInitialization().then(resolve, reject);
    });
    script.addEventListener("error", () => {
      reject(new Error(`Failed to load OpenCV.js from ${OPENCV_CDN_URL}.`));
    });
    document.head.appendChild(script);
  });

  return cvPromise;
}
