import type { ImageRect } from "../utils/roi";

const HEATMAP_SHADER = /* wgsl */ `
struct Uniforms {
  image_x: f32,
  image_y: f32,
  image_width: f32,
  image_height: f32,
  roi_x: f32,
  roi_y: f32,
  roi_width: f32,
  roi_height: f32,
  map_size: u32,
  opacity: f32,
  enabled: f32,
};

struct VertexOut {
  @builtin(position) position: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> heatmap_values: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

fn heatmap_index(x: u32, y: u32) -> u32 {
  return y * uniforms.map_size + x;
}

fn sample_heatmap(uv: vec2<f32>) -> f32 {
  let size = f32(uniforms.map_size);
  let coord = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * max(vec2<f32>(size - 1.0), vec2<f32>(0.0));
  let x0 = u32(floor(coord.x));
  let y0 = u32(floor(coord.y));
  let x1 = min(x0 + 1u, uniforms.map_size - 1u);
  let y1 = min(y0 + 1u, uniforms.map_size - 1u);
  let tx = coord.x - floor(coord.x);
  let ty = coord.y - floor(coord.y);
  let v00 = heatmap_values[heatmap_index(x0, y0)];
  let v10 = heatmap_values[heatmap_index(x1, y0)];
  let v01 = heatmap_values[heatmap_index(x0, y1)];
  let v11 = heatmap_values[heatmap_index(x1, y1)];
  let vx0 = mix(v00, v10, tx);
  let vx1 = mix(v01, v11, tx);
  return mix(vx0, vx1, ty);
}

fn colormap(value: f32) -> vec3<f32> {
  let v = clamp(value, 0.0, 1.0);
  let low = smoothstep(0.0, 0.55, v);
  let high = smoothstep(0.4, 1.0, v);
  return vec3<f32>(high, low, 0.2 + low * 0.5);
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
  if (uniforms.enabled < 0.5) {
    discard;
  }

  let x = frag_coord.x;
  let y = frag_coord.y;
  let inside_image =
    x >= uniforms.image_x &&
    y >= uniforms.image_y &&
    x <= uniforms.image_x + uniforms.image_width &&
    y <= uniforms.image_y + uniforms.image_height;
  let inside_roi =
    x >= uniforms.roi_x &&
    y >= uniforms.roi_y &&
    x <= uniforms.roi_x + uniforms.roi_width &&
    y <= uniforms.roi_y + uniforms.roi_height;

  if (!inside_image || !inside_roi) {
    discard;
  }

  let roi_uv = vec2<f32>(
    (x - uniforms.roi_x) / max(uniforms.roi_width, 1.0),
    (y - uniforms.roi_y) / max(uniforms.roi_height, 1.0)
  );
  if (uniforms.map_size == 0u) {
    discard;
  }
  let sample_value = sample_heatmap(roi_uv);
  let color = colormap(sample_value);
  return vec4<f32>(color, sample_value * uniforms.opacity);
}
`;

export class HeatmapRenderer {
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private readonly pipeline: GPURenderPipeline;
  private readonly uniformBuffer: GPUBuffer;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private heatmapBuffer: GPUBuffer;
  private bindGroup: GPUBindGroup;
  private canvasWidth = 0;
  private canvasHeight = 0;
  private mapSize = 0;

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
        module: device.createShaderModule({ code: HEATMAP_SHADER }),
        entryPoint: "vs_main",
      },
      fragment: {
        module: device.createShaderModule({ code: HEATMAP_SHADER }),
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
    this.heatmapBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.bindGroup = this.createBindGroup();
  }

  private createBindGroup(): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.heatmapBuffer } },
        { binding: 1, resource: { buffer: this.uniformBuffer } },
      ],
    });
  }

  resize(width: number, height: number): void {
    this.canvasWidth = width;
    this.canvasHeight = height;
  }

  updateHeatmap(heatmap: Float32Array, size: number): void {
    this.mapSize = size;
    const byteLength = Math.max(4, heatmap.byteLength);
    if (this.heatmapBuffer.size < byteLength) {
      this.heatmapBuffer.destroy();
      this.heatmapBuffer = this.device.createBuffer({
        size: byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.bindGroup = this.createBindGroup();
    }
    this.device.queue.writeBuffer(this.heatmapBuffer, 0, heatmap.buffer, heatmap.byteOffset, heatmap.byteLength);
  }

  getBuffer(): GPUBuffer {
    return this.heatmapBuffer;
  }

  render(params: {
    imageRect: ImageRect;
    roiRect: ImageRect | null;
    enabled: boolean;
    opacity?: number;
  }): void {
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

    if (params.roiRect && params.enabled) {
      const uniformBytes = new ArrayBuffer(48);
      const uniformView = new DataView(uniformBytes);
      uniformView.setFloat32(0, params.imageRect.x, true);
      uniformView.setFloat32(4, params.imageRect.y, true);
      uniformView.setFloat32(8, params.imageRect.width, true);
      uniformView.setFloat32(12, params.imageRect.height, true);
      uniformView.setFloat32(16, params.roiRect.x, true);
      uniformView.setFloat32(20, params.roiRect.y, true);
      uniformView.setFloat32(24, params.roiRect.width, true);
      uniformView.setFloat32(28, params.roiRect.height, true);
      uniformView.setUint32(32, this.mapSize, true);
      uniformView.setFloat32(36, params.opacity ?? 0.72, true);
      uniformView.setFloat32(40, 1, true);
      this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformBytes);
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.draw(6, 1, 0, 0);
    }

    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
