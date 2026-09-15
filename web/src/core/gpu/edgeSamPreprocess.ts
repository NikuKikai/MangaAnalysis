import { EDGE_SAM_MODEL_SIZE, type EdgeSamImageTransform, type EdgeSamPreprocessedImage } from "../onnx/edgeSamTransforms";

const CHANNEL_COUNT = 3;
const INPUT_ELEMENT_COUNT = EDGE_SAM_MODEL_SIZE * EDGE_SAM_MODEL_SIZE * CHANNEL_COUNT;

const EDGE_SAM_PREPROCESS_SHADER = /* wgsl */ `
struct Params {
  image_width: f32,
  image_height: f32,
  input_width: f32,
  input_height: f32,
  output_size: u32,
};

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output_buffer: array<f32>;

fn sample_source(output_x: f32, output_y: f32) -> vec3<f32> {
  let uv = vec2<f32>(
    (output_x + 0.5) / max(params.input_width, 1.0),
    (output_y + 0.5) / max(params.input_height, 1.0)
  );
  return textureSampleLevel(source_texture, source_sampler, uv, 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = params.output_size;
  if (id.x >= size || id.y >= size) {
    return;
  }

  let pixel_index = id.y * size + id.x;
  let plane_size = size * size;
  if (f32(id.x) >= params.input_width || f32(id.y) >= params.input_height) {
    output_buffer[pixel_index] = 0.0;
    output_buffer[plane_size + pixel_index] = 0.0;
    output_buffer[plane_size * 2u + pixel_index] = 0.0;
    return;
  }

  let rgb = sample_source(f32(id.x), f32(id.y)) * 255.0;
  output_buffer[pixel_index] = (rgb.r - 123.675) / 58.395;
  output_buffer[plane_size + pixel_index] = (rgb.g - 116.28) / 57.12;
  output_buffer[plane_size * 2u + pixel_index] = (rgb.b - 103.53) / 57.375;
}
`;

export class EdgeSamPreprocessor {
  private readonly device: GPUDevice;
  private readonly pipeline: GPUComputePipeline;
  private readonly sampler: GPUSampler;
  private readonly paramBuffer: GPUBuffer;
  private readonly outputBuffer: GPUBuffer;
  private readonly readBuffer: GPUBuffer;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private sourceTexture: GPUTexture | null = null;
  private sourceWidth = 0;
  private sourceHeight = 0;

  constructor(device: GPUDevice) {
    this.device = device;
    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      compute: {
        module: device.createShaderModule({ code: EDGE_SAM_PREPROCESS_SHADER }),
        entryPoint: "main",
      },
    });
    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    this.paramBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.outputBuffer = device.createBuffer({
      size: INPUT_ELEMENT_COUNT * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.readBuffer = device.createBuffer({
      size: INPUT_ELEMENT_COUNT * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  setSourceImage(bitmap: ImageBitmap): void {
    this.sourceWidth = bitmap.width;
    this.sourceHeight = bitmap.height;
    this.sourceTexture?.destroy();
    this.sourceTexture = this.device.createTexture({
      size: [bitmap.width, bitmap.height, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: this.sourceTexture }, [
      bitmap.width,
      bitmap.height,
      1,
    ]);
  }

  async run(): Promise<EdgeSamPreprocessedImage> {
    if (!this.sourceTexture) {
      throw new Error("EdgeSAM source texture is not ready.");
    }

    const scale = EDGE_SAM_MODEL_SIZE / Math.max(this.sourceWidth, this.sourceHeight);
    const inputWidth = Math.round(this.sourceWidth * scale);
    const inputHeight = Math.round(this.sourceHeight * scale);

    const paramsBuffer = new ArrayBuffer(32);
    const paramsView = new DataView(paramsBuffer);
    paramsView.setFloat32(0, this.sourceWidth, true);
    paramsView.setFloat32(4, this.sourceHeight, true);
    paramsView.setFloat32(8, inputWidth, true);
    paramsView.setFloat32(12, inputHeight, true);
    paramsView.setUint32(16, EDGE_SAM_MODEL_SIZE, true);
    this.device.queue.writeBuffer(this.paramBuffer, 0, paramsBuffer);

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sourceTexture.createView() },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.paramBuffer } },
        { binding: 3, resource: { buffer: this.outputBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(EDGE_SAM_MODEL_SIZE / 8), Math.ceil(EDGE_SAM_MODEL_SIZE / 8), 1);
    pass.end();
    commandEncoder.copyBufferToBuffer(
      this.outputBuffer,
      0,
      this.readBuffer,
      0,
      INPUT_ELEMENT_COUNT * Float32Array.BYTES_PER_ELEMENT,
    );
    this.device.queue.submit([commandEncoder.finish()]);

    await this.readBuffer.mapAsync(GPUMapMode.READ);
    const mapped = this.readBuffer.getMappedRange();
    const tensor = new Float32Array(mapped.slice(0));
    this.readBuffer.unmap();

    const transform: EdgeSamImageTransform = {
      originalWidth: this.sourceWidth,
      originalHeight: this.sourceHeight,
      inputWidth,
      inputHeight,
      scale,
    };

    return { tensor, transform };
  }
}
