import type { Point, RoiRect, SimulationSettings } from "../../types/simulation";

const MODEL_SIZE = 512;
const CHANNEL_COUNT = 3;
const INPUT_ELEMENT_COUNT = MODEL_SIZE * MODEL_SIZE * CHANNEL_COUNT;

const PREPROCESS_SHADER = /* wgsl */ `
struct Params {
  image_width: f32,
  image_height: f32,
  roi_x: f32,
  roi_y: f32,
  roi_size: f32,
  fixation_x: f32,
  fixation_y: f32,
  clear_radius: f32,
  max_blur_strength: f32,
  output_size: u32,
};

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output_buffer: array<f32>;

const PI: f32 = 3.141592653589793;

fn sample_page(page_x: f32, page_y: f32) -> vec3<f32> {
  if (page_x < 0.0 || page_y < 0.0 || page_x >= params.image_width || page_y >= params.image_height) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  let uv = vec2<f32>(page_x / params.image_width, page_y / params.image_height);
  return textureSampleLevel(source_texture, source_sampler, uv, 0.0).rgb;
}

fn blur_sample(page_x: f32, page_y: f32, blur_radius: f32) -> vec3<f32> {
  if (blur_radius <= 0.0001) {
    return sample_page(page_x, page_y);
  }

  // Use concentric rings with denser angular coverage to avoid directional ghosting.
  var color = sample_page(page_x, page_y) * 0.18;
  var total_weight = 0.18;
  let ring_counts = array<u32, 3>(8u, 12u, 16u);
  let ring_scales = array<f32, 3>(0.35, 0.7, 1.0);
  let ring_weights = array<f32, 3>(0.16, 0.10, 0.055);

  for (var ring = 0u; ring < 3u; ring = ring + 1u) {
    let count = ring_counts[ring];
    let radius = blur_radius * ring_scales[ring];
    let weight = ring_weights[ring];

    for (var index = 0u; index < count; index = index + 1u) {
      let angle = (2.0 * PI * (f32(index) + 0.5 * f32(ring))) / f32(count);
      let offset = vec2<f32>(cos(angle), sin(angle)) * radius;
      color = color + sample_page(page_x + offset.x, page_y + offset.y) * weight;
      total_weight = total_weight + weight;
    }
  }

  return color / max(total_weight, 0.0001);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = params.output_size;
  if (id.x >= size || id.y >= size) {
    return;
  }

  let model_x = f32(id.x) + 0.5;
  let model_y = f32(id.y) + 0.5;
  let roi_u = model_x / f32(size);
  let roi_v = model_y / f32(size);
  let page_x = params.roi_x + roi_u * params.roi_size;
  let page_y = params.roi_y + roi_v * params.roi_size;

  let dx = page_x - params.fixation_x;
  let dy = page_y - params.fixation_y;
  let distance = sqrt(dx * dx + dy * dy);
  let max_radius = max(params.roi_size * 0.5, params.clear_radius + 0.0001);
  let normalized = clamp((distance - params.clear_radius) / max(max_radius - params.clear_radius, 0.0001), 0.0, 1.0);
  let blend_factor = normalized * normalized * (3.0 - 2.0 * normalized);
  let blur_radius = blend_factor * params.max_blur_strength;

  let rgb = blur_sample(page_x, page_y, blur_radius);
  let normalized_rgb = vec3<f32>(
    (rgb.r - 0.485) / 0.229,
    (rgb.g - 0.456) / 0.224,
    (rgb.b - 0.406) / 0.225
  );

  let pixel_index = id.y * size + id.x;
  let plane_size = size * size;
  output_buffer[pixel_index] = normalized_rgb.r;
  output_buffer[plane_size + pixel_index] = normalized_rgb.g;
  output_buffer[plane_size * 2u + pixel_index] = normalized_rgb.b;
}
`;

export class RoiPreprocessor {
  private readonly device: GPUDevice;
  private readonly pipeline: GPUComputePipeline;
  private readonly sampler: GPUSampler;
  private readonly paramBuffer: GPUBuffer;
  private readonly outputBuffer: GPUBuffer;
  private readonly readBuffer: GPUBuffer;
  private sourceTexture: GPUTexture | null = null;
  private sourceWidth = 0;
  private sourceHeight = 0;
  private readonly bindGroupLayout: GPUBindGroupLayout;

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
        module: device.createShaderModule({ code: PREPROCESS_SHADER }),
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
      size: 64,
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

  async run(roi: RoiRect, fixation: Point, imageHeight: number, settings: SimulationSettings): Promise<Float32Array> {
    if (!this.sourceTexture) {
      throw new Error("Source texture is not ready.");
    }

    // WGSL uniform layout mixes floats and an unsigned integer, so pack it field-by-field.
    const paramsBuffer = new ArrayBuffer(64);
    const paramsView = new DataView(paramsBuffer);
    paramsView.setFloat32(0, this.sourceWidth, true);
    paramsView.setFloat32(4, this.sourceHeight, true);
    paramsView.setFloat32(8, roi.x, true);
    paramsView.setFloat32(12, roi.y, true);
    paramsView.setFloat32(16, roi.size, true);
    paramsView.setFloat32(20, fixation.x, true);
    paramsView.setFloat32(24, fixation.y, true);
    paramsView.setFloat32(28, imageHeight * settings.clearRadiusRatio, true);
    paramsView.setFloat32(32, settings.maxBlurStrength, true);
    paramsView.setUint32(36, MODEL_SIZE, true);
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
    pass.dispatchWorkgroups(Math.ceil(MODEL_SIZE / 8), Math.ceil(MODEL_SIZE / 8), 1);
    pass.end();
    // Read back a plain float tensor so ORT receives the exact NCHW buffer it expects.
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
    const copy = new Float32Array(mapped.slice(0));
    this.readBuffer.unmap();
    return copy;
  }
}

export function modelSize(): number {
  return MODEL_SIZE;
}
