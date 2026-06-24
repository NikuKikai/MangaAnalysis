import type { ImageRect } from "../simulation/roi";

const HISTORY_SHADER = /* wgsl */ `
struct Uniforms {
  viewport_width: f32,
  viewport_height: f32,
  image_x: f32,
  image_y: f32,
  image_width: f32,
  image_height: f32,
  map_width: u32,
  map_height: u32,
  enabled: f32,
  max_value: f32,
};

struct VertexOut {
  @builtin(position) position: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> history_map: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

fn history_index(x: u32, y: u32) -> u32 {
  return y * uniforms.map_width + x;
}

fn sample_history(uv: vec2<f32>) -> f32 {
  let size = vec2<f32>(f32(uniforms.map_width), f32(uniforms.map_height));
  let coord = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * max(size - vec2<f32>(1.0), vec2<f32>(0.0));
  let x0 = u32(floor(coord.x));
  let y0 = u32(floor(coord.y));
  let x1 = min(x0 + 1u, uniforms.map_width - 1u);
  let y1 = min(y0 + 1u, uniforms.map_height - 1u);
  let tx = coord.x - floor(coord.x);
  let ty = coord.y - floor(coord.y);
  let v00 = history_map[history_index(x0, y0)];
  let v10 = history_map[history_index(x1, y0)];
  let v01 = history_map[history_index(x0, y1)];
  let v11 = history_map[history_index(x1, y1)];
  let vx0 = mix(v00, v10, tx);
  let vx1 = mix(v01, v11, tx);
  return mix(vx0, vx1, ty);
}

fn colorize(value: f32) -> vec4<f32> {
  let v = clamp(value, 0.0, 1.0);
  let low = min(1.0, v * 1.6);
  let high = clamp((v - 0.28) / 0.72, 0.0, 1.0);
  return vec4<f32>(
    0.25 + high * 0.75,
    0.2 + low * 0.55,
    0.1 + (1.0 - high) * 0.18,
    v * (210.0 / 255.0)
  );
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
  var positions = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0)
  );
  var out: VertexOut;
  out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
  return out;
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
  if (uniforms.enabled < 0.5 || uniforms.max_value <= 0.0 || uniforms.map_width == 0u || uniforms.map_height == 0u) {
    discard;
  }

  let x = frag_coord.x;
  let y = frag_coord.y;
  let inside_image =
    x >= uniforms.image_x &&
    y >= uniforms.image_y &&
    x <= uniforms.image_x + uniforms.image_width &&
    y <= uniforms.image_y + uniforms.image_height;

  if (!inside_image) {
    discard;
  }

  let image_uv = vec2<f32>(
    (x - uniforms.image_x) / max(uniforms.image_width, 1.0),
    (y - uniforms.image_y) / max(uniforms.image_height, 1.0)
  );
  let normalized = sample_history(image_uv) / uniforms.max_value;
  let color = colorize(normalized);
  return color;
}
`;

export class HistoryRenderer {
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private readonly pipeline: GPURenderPipeline;
  private readonly uniformBuffer: GPUBuffer;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private historyBuffer: GPUBuffer;
  private bindGroup: GPUBindGroup;
  private mapWidth = 0;
  private mapHeight = 0;
  private canvasWidth = 0;
  private canvasHeight = 0;
  private maxValue = 0;

  constructor(device: GPUDevice, canvas: HTMLCanvasElement) {
    this.device = device;
    const context = canvas.getContext("webgpu");
    if (!context) {
      throw new Error("WebGPU canvas context is unavailable.");
    }
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device,
      format: this.format,
      alphaMode: "premultiplied",
    });
    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });
    this.pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      vertex: {
        module: device.createShaderModule({ code: HISTORY_SHADER }),
        entryPoint: "vs_main",
      },
      fragment: {
        module: device.createShaderModule({ code: HISTORY_SHADER }),
        entryPoint: "fs_main",
        targets: [
          {
            format: this.format,
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: {
        topology: "triangle-list",
      },
    });
    this.uniformBuffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.historyBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.bindGroup = this.createBindGroup();
  }

  private createBindGroup(): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.historyBuffer } },
        { binding: 1, resource: { buffer: this.uniformBuffer } },
      ],
    });
  }

  resize(width: number, height: number): void {
    this.canvasWidth = width;
    this.canvasHeight = height;
  }

  updateHistory(historyMap: Float32Array | null, width: number, height: number): void {
    this.mapWidth = width;
    this.mapHeight = height;
    this.maxValue = 0;

    if (!historyMap || width <= 0 || height <= 0) {
      return;
    }

    for (let index = 0; index < historyMap.length; index += 1) {
      if (historyMap[index] > this.maxValue) {
        this.maxValue = historyMap[index];
      }
    }

    const byteLength = Math.max(4, historyMap.byteLength);
    if (this.historyBuffer.size < byteLength) {
      this.historyBuffer.destroy();
      this.historyBuffer = this.device.createBuffer({
        size: byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.bindGroup = this.createBindGroup();
    }
    this.device.queue.writeBuffer(this.historyBuffer, 0, historyMap);
  }

  getBuffer(): GPUBuffer {
    return this.historyBuffer;
  }

  render(params: { imageRect: ImageRect | null; enabled: boolean }): void {
    const textureView = this.context.getCurrentTexture().createView();
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    if (params.imageRect && params.enabled) {
      const uniforms = new ArrayBuffer(64);
      const view = new DataView(uniforms);
      view.setFloat32(0, this.canvasWidth, true);
      view.setFloat32(4, this.canvasHeight, true);
      view.setFloat32(8, params.imageRect.x, true);
      view.setFloat32(12, params.imageRect.y, true);
      view.setFloat32(16, params.imageRect.width, true);
      view.setFloat32(20, params.imageRect.height, true);
      view.setUint32(24, this.mapWidth, true);
      view.setUint32(28, this.mapHeight, true);
      view.setFloat32(32, params.enabled ? 1 : 0, true);
      view.setFloat32(36, this.maxValue, true);
      this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.draw(6, 1, 0, 0);
    }

    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
