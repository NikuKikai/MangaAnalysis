from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np
import torch
import wgpu
from PIL import Image
from wgpu.utils import get_default_device

_PREPROCESS_SHADER = """
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

  // Use concentric rings to approximate the browser-side radial blur kernel.
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
"""


@dataclass(slots=True)
class RoiWindow:
    """Square ROI cut from the page, including black padding outside page bounds."""

    image: Image.Image
    fixation_x: float
    fixation_y: float
    origin_x: int
    origin_y: int
    crop_left: int
    crop_top: int
    crop_right: int
    crop_bottom: int
    page_width: int
    page_height: int

    @property
    def fixation_in_roi(self) -> tuple[float, float]:
        """Return the fixation position expressed in local ROI coordinates."""
        return float(self.fixation_x - self.crop_left + self.origin_x), float(self.fixation_y - self.crop_top + self.origin_y)


@dataclass(slots=True)
class WgpuRoiPreprocessor:
    """GPU ROI preprocessor that mirrors the web-side WGPU shader path."""

    model_size: int
    device: wgpu.GPUDevice = field(init=False, repr=False)
    bind_group_layout: wgpu.GPUBindGroupLayout = field(init=False, repr=False)
    pipeline: wgpu.GPUComputePipeline = field(init=False, repr=False)
    sampler: wgpu.GPUSampler = field(init=False, repr=False)
    param_buffer: wgpu.GPUBuffer = field(init=False, repr=False)
    output_buffer: wgpu.GPUBuffer = field(init=False, repr=False)
    source_texture: wgpu.GPUTexture | None = field(init=False, default=None, repr=False)
    source_image_id: int | None = field(init=False, default=None, repr=False)
    source_size: tuple[int, int] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Create GPU resources that are reused across all ROI evaluations."""
        self.device = get_default_device()
        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "texture": {"sample_type": wgpu.TextureSampleType.float},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                },
            ]
        )
        self.pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[self.bind_group_layout]),
            compute={
                "module": self.device.create_shader_module(code=_PREPROCESS_SHADER),
                "entry_point": "main",
            },
        )
        self.sampler = self.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            mipmap_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
        )
        self.param_buffer = self.device.create_buffer(
            size=64,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.output_buffer = self.device.create_buffer(
            size=self._tensor_byte_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

    @property
    def _tensor_byte_size(self) -> int:
        """Return the byte size of one model input tensor."""
        return int(self.model_size * self.model_size * 3 * np.dtype(np.float32).itemsize)

    def preprocess(
        self,
        page_image: Image.Image,
        roi_window: RoiWindow,
        clear_radius: float,
        max_blur_strength: float,
    ) -> torch.Tensor:
        """Run the web-aligned GPU preprocess path and return a normalized NCHW tensor."""
        self._ensure_source_texture(page_image)
        roi_left = float(roi_window.crop_left - roi_window.origin_x)
        roi_top = float(roi_window.crop_top - roi_window.origin_y)
        roi_size = float(roi_window.image.size[0])

        # Pack WGSL uniforms in field order to keep the Python layout identical to the web version.
        params = struct.pack(
            "<9fI",
            float(page_image.size[0]),
            float(page_image.size[1]),
            roi_left,
            roi_top,
            roi_size,
            float(roi_window.fixation_x),
            float(roi_window.fixation_y),
            float(clear_radius),
            float(max_blur_strength),
            int(self.model_size),
        )
        self.device.queue.write_buffer(self.param_buffer, 0, params + b"\x00" * (64 - len(params)))

        bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.source_texture.create_view()},
                {"binding": 1, "resource": self.sampler},
                {"binding": 2, "resource": {"buffer": self.param_buffer}},
                {"binding": 3, "resource": {"buffer": self.output_buffer}},
            ],
        )

        # Dispatch the compute shader and read back the plain float tensor for PyTorch inference.
        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        workgroup_count = (self.model_size + 7) // 8
        compute_pass.dispatch_workgroups(workgroup_count, workgroup_count, 1)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])

        tensor_view = self.device.queue.read_buffer(self.output_buffer)
        tensor_array = np.frombuffer(tensor_view, dtype=np.float32).copy()
        return torch.from_numpy(tensor_array.reshape(1, 3, self.model_size, self.model_size))

    def _ensure_source_texture(self, page_image: Image.Image) -> None:
        """Upload the source page once and reuse its texture across ROI evaluations."""
        page_size = tuple(page_image.size)
        if self.source_texture is not None and self.source_image_id == id(page_image) and self.source_size == page_size:
            return

        rgba = np.asarray(page_image.convert("RGBA"), dtype=np.uint8)
        self.source_texture = self.device.create_texture(
            size=(page_size[0], page_size[1], 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self.device.queue.write_texture(
            {"texture": self.source_texture},
            rgba,
            {"bytes_per_row": page_size[0] * 4, "rows_per_image": page_size[1]},
            (page_size[0], page_size[1], 1),
        )
        self.source_image_id = id(page_image)
        self.source_size = page_size


def extract_roi(page_image: Image.Image, fixation: tuple[float, float], roi_size: int) -> RoiWindow:
    """Extract a fixation-centered square ROI and pad out-of-page regions with black."""
    page_width, page_height = page_image.size
    center_x = int(round(fixation[0]))
    center_y = int(round(fixation[1]))
    half = roi_size // 2

    left = center_x - half
    top = center_y - half
    right = left + roi_size
    bottom = top + roi_size

    crop_left = max(0, left)
    crop_top = max(0, top)
    crop_right = min(page_width, right)
    crop_bottom = min(page_height, bottom)

    cropped = page_image.crop((crop_left, crop_top, crop_right, crop_bottom))
    roi = Image.new("RGB", (roi_size, roi_size), (0, 0, 0))
    origin_x = crop_left - left
    origin_y = crop_top - top
    roi.paste(cropped, (origin_x, origin_y))

    return RoiWindow(
        image=roi,
        fixation_x=float(fixation[0]),
        fixation_y=float(fixation[1]),
        origin_x=origin_x,
        origin_y=origin_y,
        crop_left=crop_left,
        crop_top=crop_top,
        crop_right=crop_right,
        crop_bottom=crop_bottom,
        page_width=page_width,
        page_height=page_height,
    )
