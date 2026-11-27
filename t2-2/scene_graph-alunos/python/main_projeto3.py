from __future__ import annotations

import time
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

# amostras por pixel do passe principal (MSAA); Renderer e todos os
# pipelines desse passe precisam usar o mesmo valor
SAMPLE_COUNT = 4

# Medicao do tempo de render. O FPS da janela nao serve: no macOS o compositor
# so entrega a textura da tela a 60 Hz (get_current_texture bloqueia), mesmo sem vsync.
render_ms_sum = 0.0
render_frames = 0
render_t_last = time.perf_counter()


def make_trf (tx: float, ty: float, tz: float,
              sx: float = 1.0, sy: float = 1.0, sz: float = 1.0,
              ry: float = 0.0) -> Transform:
  trf = Transform()
  trf.translate(tx, ty, tz)
  if ry != 0.0:
    trf.rotate(ry, 0.0, 1.0, 0.0)
  trf.scale(sx, sy, sz)
  return trf


def initialize (device: wgpu.GPUDevice, target_format: str) -> None:
  global camera, scene

  camera = Camera3D(4.5, 4.0, 7.0)
  camera.set_center(0.0, 0.8, 0.0)
  arcball = camera.create_arcball()
  arcball.attach(canvas)

  light = PointLight(-2.5, 4.0, 2.5, space="world")

  cinza    = PhongMaterial(0.70, 0.70, 0.70)
  amarelo  = PhongMaterial(0.90, 0.65, 0.10)
  vermelho = PhongMaterial(0.90, 0.20, 0.20)
  verde    = PhongMaterial(0.25, 0.75, 0.35)
  for mat in (cinza, amarelo, vermelho, verde):
    mat.set_shininess(64.0)

  cube = Cube(device)
  sphere = Sphere(device)

  shader = Shader(device, "../shaders/ilum_frag/cena.wgsl", light=light, space="world")
  shader.set_vertex_buffers([
    {"array_stride": 3 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x3", "offset": 0, "var_name": "coord"}]},
    {"array_stride": 3 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x3", "offset": 0, "var_name": "normal"}]},
  ])
  pipeline = Pipeline(shader, target_format, primitive={"cull_mode": "back"},
                      multisample={"count": SAMPLE_COUNT})
  for mat in (cinza, amarelo, vermelho, verde):
    shader.add_material(mat)

  chao = Node(None, make_trf(0.0, -0.2, 0.0, 10.0, 0.2, 10.0), [cinza], [cube])
  caixa = Node(None, make_trf(-1.2, 0.0, 0.2, ry=25.0), [amarelo], [cube])
  esfera_vermelha = Node(None, make_trf(1.4, 0.6, -0.4, 0.6, 0.6, 0.6), [vermelho], [sphere])
  esfera_verde = Node(None, make_trf(0.2, 0.3, 1.5, 0.3, 0.3, 0.3), [verde], [sphere])

  root = Node(pipeline, nodes=[chao, caixa, esfera_vermelha, esfera_verde])
  scene = Scene(root)


def draw () -> None:
  global render_ms_sum, render_frames, render_t_last
  target_texture = context.get_current_texture()

  # cronometra so o passe: comeca depois da espera pela textura e
  # termina quando a GPU conclui o trabalho submetido
  t0 = time.perf_counter()
  renderer.render(target_texture, scene, camera)
  device.queue.on_submitted_work_done_sync()
  render_ms_sum += (time.perf_counter() - t0) * 1000.0
  render_frames += 1

  # media de ~1 s no titulo, com a resolucao em pixels fisicos
  if time.perf_counter() - render_t_last >= 1.0:
    canvas.set_title(f"Projeto 3 - $fps fps | render {render_ms_sum / render_frames:.2f} ms"
                     f" | {target_texture.width}x{target_texture.height}")
    render_ms_sum, render_frames = 0.0, 0
    render_t_last = time.perf_counter()


def on_key (event: Any) -> None:
  if event["key"] == "q":
    canvas.close()


def main () -> None:
  global canvas, device, context, renderer

  # $fps e $ms sao substituidos pelo rendercanvas a cada quadro (analise de desempenho)
  canvas = RenderCanvas(size=(800, 600), title="Projeto 3 - $fps fps ($ms ms)",
                        update_mode="continuous", max_fps=60)
  adapter = wgpu.gpu.request_adapter_sync()
  device = adapter.request_device_sync()
  context = canvas.get_context("wgpu")
  target_format = context.get_preferred_format(device.adapter)
  context.configure(device=device, format=target_format)

  renderer = Renderer(device, depth_test=True, clear_value=(0.9, 0.9, 0.9, 1.0),
                      sample_count=SAMPLE_COUNT)

  initialize(device, target_format)

  canvas.add_event_handler(on_key, "key_down")
  canvas.request_draw(draw)
  loop.run()


if __name__ == "__main__":
  main()