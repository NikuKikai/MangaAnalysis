import * as ort from "onnxruntime-web";

const MODEL_SIZE = 512;
const fp16ModelUrl = `${import.meta.env.BASE_URL}models/stage1_salicon_pretrained_512_v2w075_sigmoid_fp16.onnx`;
const ortWasmJsepUrl = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort-wasm-simd-threaded.jsep.wasm";

export class SaliencySession {
  private constructor(private readonly session: ort.InferenceSession) {}

  static async create(): Promise<SaliencySession> {
    ort.env.wasm.proxy = false;
    ort.env.wasm.numThreads = 1;
    // In dev, ORT may infer a fallback HTML URL instead of the wasm asset, so pin it explicitly.
    ort.env.wasm.wasmPaths = {
      wasm: ortWasmJsepUrl,
    };
    ort.env.webgpu.powerPreference = "high-performance";
    const options: ort.InferenceSession.SessionOptions = {
      executionProviders: [
        {
          name: "webgpu",
          // The exported saliency model uses NCHW tensors end-to-end.
          preferredLayout: "NCHW",
        },
        "wasm",
      ],
      // Full graph fusion can rewrite float16 conv blocks into kernels that ORT Web later
      // tries to place on the JS EP, which does not support float16 FusedConv.
      graphOptimizationLevel: "basic",
    };

    try {
      const session = await ort.InferenceSession.create(fp16ModelUrl, options);
      return new SaliencySession(session);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (
        message.includes("no available backend found") ||
        message.includes("initWasm") ||
        message.includes("BigInt")
      ) {
        throw new Error(`Failed to initialize ONNX Runtime backend. ${message}`);
      }
      throw new Error(`Failed to load FP16 saliency model. ${message}`);
    }
  }

  async predict(input: Float32Array): Promise<Float32Array> {
    const tensor = new ort.Tensor("float32", input, [1, 3, MODEL_SIZE, MODEL_SIZE]);
    const output = await this.session.run({ input_image: tensor });
    const saliency = output.saliency_map.data;
    return saliency instanceof Float32Array ? saliency : Float32Array.from(saliency);
  }
}
