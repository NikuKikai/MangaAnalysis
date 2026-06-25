import type { Candidate, Point, RoiRect } from "../../types/simulation";

const BYTES_PER_CANDIDATE = 36;

const CANDIDATE_SHADER = /* wgsl */ `
struct Params {
  map_size: u32,
  image_width: u32,
  image_height: u32,
  history_width: u32,
  history_height: u32,
  nms_radius: u32,
  roi_x: f32,
  roi_y: f32,
  roi_size: f32,
  fixation_x: f32,
  fixation_y: f32,
  history_alpha: f32,
  distance_sigma: f32,
};

struct CandidateData {
  model_x: u32,
  model_y: u32,
  page_x: f32,
  page_y: f32,
  heatmap_value: f32,
  history_value: f32,
  inhibition_score: f32,
  distance_score: f32,
  final_score: f32,
};

struct CandidateBuffer {
  count: atomic<u32>,
  padding0: u32,
  padding1: u32,
  padding2: u32,
  entries: array<CandidateData>,
};

@group(0) @binding(0) var<storage, read> heatmap_values: array<f32>;
@group(0) @binding(1) var<storage, read> history_values: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output_candidates: CandidateBuffer;

fn heatmap_index(x: u32, y: u32) -> u32 {
  return y * params.map_size + x;
}

fn history_index(x: u32, y: u32) -> u32 {
  return y * params.history_width + x;
}

fn distance_score(from_x: f32, from_y: f32, to_x: f32, to_y: f32, sigma: f32) -> f32 {
  let dx = to_x - from_x;
  let dy = to_y - from_y;
  let sigma_sq = max(sigma * sigma, 1e-6);
  return exp(-0.5 * (dx * dx + dy * dy) / sigma_sq);
}

fn final_score(page_x: f32, page_y: f32, heatmap_value: f32) -> f32 {
  let history_value = sample_history(page_x, page_y);
  let inhibition_score = exp(-params.history_alpha * history_value);
  let distance_weight = distance_score(params.fixation_x, params.fixation_y, page_x, page_y, params.distance_sigma);
  return heatmap_value * inhibition_score * distance_weight;
}

fn sample_history(page_x: f32, page_y: f32) -> f32 {
  if (params.history_width == 0u || params.history_height == 0u) {
    return 0.0;
  }
  let px = clamp(u32(round(page_x)), 0u, params.history_width - 1u);
  let py = clamp(u32(round(page_y)), 0u, params.history_height - 1u);
  return history_values[history_index(px, py)];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.map_size || id.y >= params.map_size) {
    return;
  }

  let index = heatmap_index(id.x, id.y);
  let heatmap_value = heatmap_values[index];

  let page_x = params.roi_x + ((f32(id.x) + 0.5) / f32(params.map_size)) * params.roi_size;
  let page_y = params.roi_y + ((f32(id.y) + 0.5) / f32(params.map_size)) * params.roi_size;
  if (page_x < 0.0 || page_y < 0.0 || page_x >= f32(params.image_width) || page_y >= f32(params.image_height)) {
    return;
  }

  let history_value = sample_history(page_x, page_y);
  let inhibition_score = exp(-params.history_alpha * history_value);
  let distance_weight = distance_score(params.fixation_x, params.fixation_y, page_x, page_y, params.distance_sigma);
  let candidate_final_score = final_score(page_x, page_y, heatmap_value);

  let radius = i32(params.nms_radius);
  for (var dy = -radius; dy <= radius; dy = dy + 1) {
    for (var dx = -radius; dx <= radius; dx = dx + 1) {
      let nx = i32(id.x) + dx;
      let ny = i32(id.y) + dy;
      if (nx < 0 || ny < 0 || nx >= i32(params.map_size) || ny >= i32(params.map_size)) {
        continue;
      }

      let neighbor_page_x = params.roi_x + ((f32(nx) + 0.5) / f32(params.map_size)) * params.roi_size;
      let neighbor_page_y = params.roi_y + ((f32(ny) + 0.5) / f32(params.map_size)) * params.roi_size;
      if (
        neighbor_page_x < 0.0 ||
        neighbor_page_y < 0.0 ||
        neighbor_page_x >= f32(params.image_width) ||
        neighbor_page_y >= f32(params.image_height)
      ) {
        continue;
      }

      let neighbor_heatmap_value = heatmap_values[heatmap_index(u32(nx), u32(ny))];
      let neighbor_final_score = final_score(neighbor_page_x, neighbor_page_y, neighbor_heatmap_value);
      if (neighbor_final_score > candidate_final_score) {
        return;
      }
    }
  }

  let offset = atomicAdd(&output_candidates.count, 1u);
  if (offset >= params.map_size * params.map_size) {
    return;
  }
  output_candidates.entries[offset] = CandidateData(
    id.x,
    id.y,
    page_x,
    page_y,
    heatmap_value,
    history_value,
    inhibition_score,
    distance_weight,
    candidate_final_score
  );
}
`;

type CandidateRecord = {
  modelX: number;
  modelY: number;
  pageX: number;
  pageY: number;
  saliencyScore: number;
  historyValue: number;
  inhibitionScore: number;
  distanceScore: number;
  finalScore: number;
};

export class GpuCandidateSelector {
  private readonly device: GPUDevice;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private readonly paramBuffer: GPUBuffer;
  private outputBuffer: GPUBuffer | null = null;
  private readBuffer: GPUBuffer | null = null;
  private outputCapacity = 0;

  constructor(device: GPUDevice) {
    this.device = device;
    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      compute: {
        module: device.createShaderModule({ code: CANDIDATE_SHADER }),
        entryPoint: "main",
      },
    });
    this.paramBuffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  private ensureBuffers(mapSize: number): void {
    const requiredCapacity = mapSize * mapSize;
    if (requiredCapacity <= this.outputCapacity && this.outputBuffer && this.readBuffer) {
      return;
    }

    this.outputBuffer?.destroy();
    this.readBuffer?.destroy();
    this.outputCapacity = requiredCapacity;
    const bufferSize = 16 + this.outputCapacity * BYTES_PER_CANDIDATE;
    this.outputBuffer = this.device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.readBuffer = this.device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  async select(params: {
    heatmapBuffer: GPUBuffer;
    historyBuffer: GPUBuffer;
    mapSize: number;
    nmsRadius: number;
    topK: number;
    roi: RoiRect;
    imageWidth: number;
    imageHeight: number;
    currentFixation: Point;
    historyMapWidth: number;
    historyMapHeight: number;
    historyAlpha: number;
    distanceSigma: number;
  }): Promise<Candidate[]> {
    const {
      heatmapBuffer,
      historyBuffer,
      mapSize,
      nmsRadius,
      topK,
      roi,
      imageWidth,
      imageHeight,
      currentFixation,
      historyMapWidth,
      historyMapHeight,
      historyAlpha,
      distanceSigma,
    } = params;

    this.ensureBuffers(mapSize);
    if (!this.outputBuffer || !this.readBuffer) {
      throw new Error("Candidate buffers are not initialized.");
    }

    this.device.queue.writeBuffer(this.outputBuffer, 0, new Uint32Array([0, 0, 0, 0]));
    const uniformBytes = new ArrayBuffer(64);
    const view = new DataView(uniformBytes);
    view.setUint32(0, mapSize, true);
    view.setUint32(4, imageWidth, true);
    view.setUint32(8, imageHeight, true);
    view.setUint32(12, historyMapWidth, true);
    view.setUint32(16, historyMapHeight, true);
    view.setUint32(20, nmsRadius, true);
    view.setFloat32(24, roi.x, true);
    view.setFloat32(28, roi.y, true);
    view.setFloat32(32, roi.size, true);
    view.setFloat32(36, currentFixation.x, true);
    view.setFloat32(40, currentFixation.y, true);
    view.setFloat32(44, historyAlpha, true);
    view.setFloat32(48, distanceSigma, true);
    this.device.queue.writeBuffer(this.paramBuffer, 0, uniformBytes);

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: heatmapBuffer } },
        { binding: 1, resource: { buffer: historyBuffer } },
        { binding: 2, resource: { buffer: this.paramBuffer } },
        { binding: 3, resource: { buffer: this.outputBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(mapSize / 8), Math.ceil(mapSize / 8), 1);
    pass.end();
    encoder.copyBufferToBuffer(this.outputBuffer, 0, this.readBuffer, 0, 16 + this.outputCapacity * BYTES_PER_CANDIDATE);
    this.device.queue.submit([encoder.finish()]);

    await this.readBuffer.mapAsync(GPUMapMode.READ);
    const bytes = this.readBuffer.getMappedRange();
    const raw = new DataView(bytes.slice(0));
    const count = Math.min(raw.getUint32(0, true), this.outputCapacity);
    const candidates: CandidateRecord[] = [];
    for (let index = 0; index < count; index += 1) {
      const base = 16 + index * BYTES_PER_CANDIDATE;
      const heatmapValue = raw.getFloat32(base + 16, true);
      const historyValue = raw.getFloat32(base + 20, true);
      const inhibitionScore = raw.getFloat32(base + 24, true);
      const distanceWeight = raw.getFloat32(base + 28, true);
      const finalScore = raw.getFloat32(base + 32, true);
      candidates.push({
        modelX: raw.getUint32(base + 0, true),
        modelY: raw.getUint32(base + 4, true),
        pageX: raw.getFloat32(base + 8, true),
        pageY: raw.getFloat32(base + 12, true),
        saliencyScore: heatmapValue,
        historyValue,
        inhibitionScore,
        distanceScore: distanceWeight,
        finalScore,
      });
    }
    this.readBuffer.unmap();

    candidates.sort((left, right) => right.finalScore - left.finalScore);
    return candidates.slice(0, topK);
  }
}
