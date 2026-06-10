declare module "onnxruntime-web" {
  export type TypedTensorData =
    | Float32Array
    | Uint8Array
    | Int8Array
    | Uint16Array
    | Int16Array
    | Uint32Array
    | Int32Array;

  export class Tensor<T extends TypedTensorData = TypedTensorData> {
    constructor(type: string, data: T, dims: readonly number[]);
    data: T;
    dims: readonly number[];
  }

  export namespace env {
    namespace wasm {
      let proxy: boolean;
      let numThreads: number;
      let wasmPaths: {
        wasm?: string;
        [key: string]: string | undefined;
      };
    }

    namespace webgpu {
      let powerPreference: GPURequestAdapterOptions["powerPreference"];
    }
  }

  export namespace InferenceSession {
    export type ExecutionProviderConfig =
      | "wasm"
      | {
          name: "webgpu";
          preferredLayout?: "NCHW" | "NHWC";
        };

    export interface SessionOptions {
      executionProviders?: ExecutionProviderConfig[];
      graphOptimizationLevel?: "disabled" | "basic" | "extended" | "all";
    }

    export function create(model: string | ArrayBufferLike, options?: SessionOptions): Promise<InferenceSession>;
  }

  export interface InferenceSession {
    run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  }
}
