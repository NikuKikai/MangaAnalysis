import type { ImageRect } from "../simulation/roi";

const HEATMAP_SHADER = /* wgsl */ `
struct Uniforms {
  viewport_width: f32,
  viewport_height: f32,
  image_x: f32,
  image_y: f32,
  image_width: f32,
  image_height: f32,
  roi_x: f32,
  roi_y: f32,
  roi_width: f32,
  roi_height: f32,
  opacity: f32,
  enabled: f32,
};

struct VertexOut {
  @builtin(position) position: vec4<f32>,
};

@group(0) @binding(0) var heatmap_texture: texture_2d<f32>;
@group(0) @binding(1) var heatmap_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

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
  let sample_value = textureSampleLevel(heatmap_texture, heatmap_sampler, roi_uv, 0.0).r;
  let color = colormap(sample_value);
  return vec4<f32>(color, sample_value * uniforms.opacity);
}
`;

export class HeatmapRenderer {
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private readonly pipeline: GPURenderPipeline;
  private readonly sampler: GPUSampler;
  private readonly uniformBuffer: GPUBuffer;
  private heatmapTexture: GPUTexture | null = null;
  private heatmapView: GPUTextureView | null = null;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private canvasWidth = 0;
  private canvasHeight = 0;

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
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
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
    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    this.uniformBuffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  resize(width: number, height: number): void {
    this.canvasWidth = width;
    this.canvasHeight = height;
  }

  updateHeatmap(heatmap: Float32Array, size: number): void {
    const rgba = new Uint8Array(size * size * 4);
    for (let index = 0; index < heatmap.length; index += 1) {
      const value = Math.max(0, Math.min(255, Math.round(heatmap[index] * 255)));
      const base = index * 4;
      rgba[base] = value;
      rgba[base + 1] = value;
      rgba[base + 2] = value;
      rgba[base + 3] = 255;
    }
    this.heatmapTexture?.destroy();
    this.heatmapTexture = this.device.createTexture({
      size: [size, size, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.writeTexture(
      { texture: this.heatmapTexture },
      rgba,
      { bytesPerRow: size * 4 },
      { width: size, height: size, depthOrArrayLayers: 1 },
    );
    this.heatmapView = this.heatmapTexture.createView();
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

    if (this.heatmapView && params.roiRect && params.enabled) {
      const uniforms = new Float32Array([
        this.canvasWidth,
        this.canvasHeight,
        params.imageRect.x,
        params.imageRect.y,
        params.imageRect.width,
        params.imageRect.height,
        params.roiRect.x,
        params.roiRect.y,
        params.roiRect.width,
        params.roiRect.height,
        params.opacity ?? 0.72,
        1,
      ]);
      this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);
      const bindGroup = this.device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: this.heatmapView },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: { buffer: this.uniformBuffer } },
        ],
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(6, 1, 0, 0);
    }

    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
