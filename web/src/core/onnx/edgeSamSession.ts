import * as ort from "onnxruntime-web";

import {
  EDGE_SAM_MODEL_SIZE,
  postprocessEdgeSamMasks,
  transformEdgeSamPoints,
  type EdgeSamImageTransform,
  type EdgeSamPreprocessedImage,
} from "./edgeSamTransforms";
import type { EdgeSamPrediction, EdgeSamPrompt } from "./edgeSamTypes";

const encoderModelUrl = `${import.meta.env.BASE_URL}models/edge_sam_3x_encoder.onnx`;
const decoderModelUrl = `${import.meta.env.BASE_URL}models/edge_sam_3x_decoder.onnx`;
const ortWasmJsepUrl = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort-wasm-simd-threaded.jsep.wasm";

export class EdgeSamSession {
  private imageEmbedding: Float32Array | null = null;
  private imageTransform: EdgeSamImageTransform | null = null;

  private constructor(
    private readonly encoder: ort.InferenceSession,
    private readonly decoder: ort.InferenceSession,
  ) {}

  static async create(): Promise<EdgeSamSession> {
    ort.env.wasm.proxy = false;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = {
      wasm: ortWasmJsepUrl,
    };

    const options: ort.InferenceSession.SessionOptions = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "basic",
    };

    try {
      const [encoder, decoder] = await Promise.all([
        ort.InferenceSession.create(encoderModelUrl, options),
        ort.InferenceSession.create(decoderModelUrl, options),
      ]);
      return new EdgeSamSession(encoder, decoder);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to load EdgeSAM ONNX models. ${message}`);
    }
  }

  async setImage(input: EdgeSamPreprocessedImage): Promise<void> {
    const { tensor, transform } = input;
    const image = new ort.Tensor("float32", tensor, [1, 3, EDGE_SAM_MODEL_SIZE, EDGE_SAM_MODEL_SIZE]);
    const output = await this.encoder.run({ image });
    const embedding = output.image_embeddings?.data ?? Object.values(output)[0]?.data;

    if (!(embedding instanceof Float32Array)) {
      throw new Error("EdgeSAM encoder did not return a float32 image embedding.");
    }

    this.imageEmbedding = embedding;
    this.imageTransform = transform;
  }

  async predict(prompt: EdgeSamPrompt): Promise<EdgeSamPrediction> {
    if (!this.imageEmbedding || !this.imageTransform) {
      throw new Error("Call setImage(...) before running EdgeSAM prediction.");
    }
    if (prompt.points.length === 0) {
      throw new Error("EdgeSAM prediction requires at least one point prompt.");
    }

    const { coords, labels } = transformEdgeSamPoints(prompt.points, this.imageTransform);
    const output = await this.decoder.run({
      image_embeddings: new ort.Tensor("float32", this.imageEmbedding, [1, 256, 64, 64]),
      point_coords: new ort.Tensor("float32", coords, [1, prompt.points.length, 2]),
      point_labels: new ort.Tensor("float32", labels, [1, prompt.points.length]),
    });

    const scores = output.scores?.data ?? Object.values(output)[0]?.data;
    const lowResLogits = output.masks?.data ?? Object.values(output)[1]?.data;
    const lowResMaskUint8 = output.masks_uint8?.data ?? Object.values(output)[2]?.data;

    if (!(scores instanceof Float32Array) || !(lowResLogits instanceof Float32Array) || !(lowResMaskUint8 instanceof Uint8Array)) {
      throw new Error("EdgeSAM decoder did not return float32 scores/logits and uint8 masks.");
    }

    return {
      masks: postprocessEdgeSamMasks(lowResMaskUint8, scores, this.imageTransform),
      lowResLogits,
      lowResMaskUint8,
    };
  }

  clearImage(): void {
    this.imageEmbedding = null;
    this.imageTransform = null;
  }
}
