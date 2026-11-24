from __future__ import annotations

from typing import Any

import graphics_math as gm
import wgpu
from rendercanvas.glfw import RenderCanvas, loop

from camera3d import *
from pointlight import *
from shader import *
from pipeline import *
from phongmaterial import *
from transform import *
from node import *
from scene import *
from renderer import *
from cube import *
from sphere import *

canvas: RenderCanvas
device: wgpu.GPUDevice
context: Any
renderer: Renderer
camera: Camera3D
scene: Scene


def make_trf (tx: float, ty: float, tz: float,
              sx: float = 1.0, sy: float = 1.0, sz: float = 1.0) -> Transform:
  trf = Transform()
  trf.translate(tx, ty, tz)
  trf.scale(sx, sy, sz)
  return trf


def initialize (device: wgpu.GPUDevice, target_format: str) -> None:
  global camera, scene

  camera = Camera3D(1.5, 2.5, 3.0)
  camera.set_center(0.0, 0.8, 0.0)
  arcball = camera.create_arcball()
  arcball.attach(canvas)

  light = PointLight(2.0, 4.0, 3.0, space="world")

  white  = PhongMaterial(0.95, 0.95, 0.95)
  yellow = PhongMaterial(0.90, 0.65, 0.10)
  green  = PhongMaterial(0.55, 0.85, 0.45)
  red    = PhongMaterial(1.00, 0.20, 0.20)

  cube = Cube(device)
  sphere = Sphere(device)

  shader = Shader(device, "../shaders/ilum_frag/lit.wgsl", light=light, space="camera")
  shader.set_vertex_buffers([
    {"array_stride": 3 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x3", "offset": 0, "var_name": "coord"}]},
    {"array_stride": 3 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x3", "offset": 0, "var_name": "normal"}]},
  ])
  pipeline = Pipeline(shader, target_format, primitive={"cull_mode": "back"})
  for mat in (white, yellow, green, red):
    shader.add_material(mat)

  leg_h, top_h = 0.9, 0.1
  legs = [Node(None, make_trf(x, 0.0, z, 0.1, leg_h, 0.1), None, [cube])
          for x in (-0.9, 0.9) for z in (-0.6, 0.6)]
  top = Node(None, make_trf(0.0, leg_h, 0.0, 2.0, top_h, 1.4), None, [cube])
  table = Node(None, None, [white], nodes=[top] + legs)

  box_h, r_green = 0.25, 0.15
  box_shape = Node(None, make_trf(0.0, 0.0, 0.0, 0.6, box_h, 0.5), [yellow], [cube])
  green_ball = Node(None, make_trf(0.0, box_h + r_green, 0.0, r_green, r_green, r_green),
                    [green], [sphere])
  box = Node(None, make_trf(-0.2, 1.0, 0.0), nodes=[box_shape, green_ball])

  r_red = 0.3
  red_ball = Node(None, make_trf(0.55, 1.0 + r_red, -0.35, r_red, r_red, r_red),
                  [red], [sphere])

  root = Node(pipeline, nodes=[table, box, red_ball])
  scene = Scene(root)


def draw () -> None:
  target_texture = context.get_current_texture()
  renderer.render(target_texture, scene, camera)


def on_key (event: Any) -> None:
  if event["key"] == "q":
    canvas.close()


def main () -> None:
  global canvas, device, context, renderer

  canvas = RenderCanvas(size=(800, 600), title="Tarefa 2.1", update_mode="continuous", max_fps=60)
  adapter = wgpu.gpu.request_adapter_sync()
  device = adapter.request_device_sync()
  context = canvas.get_context("wgpu")
  target_format = context.get_preferred_format(device.adapter)
  context.configure(device=device, format=target_format)

  renderer = Renderer(device, depth_test=True, clear_value=(1.0, 1.0, 1.0, 1.0))

  initialize(device, target_format)

  canvas.add_event_handler(on_key, "key_down")
  canvas.request_draw(draw)
  loop.run()


if __name__ == "__main__":
  main()
