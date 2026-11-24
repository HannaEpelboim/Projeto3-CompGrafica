from __future__ import annotations

import math
from typing import Any

import graphics_math as gm

from light import Light


class SpotLight(Light):
    """Spot light: position (translated by the mapping), direction
    (ignores translation and is normalized), opening angle and
    concentration exponent. The opening angle is sent to the shader as
    its cosine, so the fragment stage compares c >= cos(theta) directly."""

    def __init__ (self, x: float, y: float, z: float,
                  dx: float, dy: float, dz: float,
                  angle: float, exponent: float,
                  space: str = "world", **values: Any) -> None:
      Light.__init__(self, space, **values)
      self.set_position(x, y, z)
      self.set_direction(dx, dy, dz)
      self.set_angle(angle)
      self.set_exponent(exponent)

    def set_position (self, x: float, y: float, z: float) -> None:
      self.set("light_position", gm.vec4(x,y,z,1.0))

    def set_direction (self, x: float, y: float, z: float) -> None:
      self.set("spot_direction", gm.vec3(x,y,z))

    def set_angle (self, angle: float) -> None:
      """Opening angle theta, in degrees (measured from the central axis)."""
      self.set("spot_cos_cutoff", math.cos(gm.radians(angle)))

    def set_exponent (self, exponent: float) -> None:
      self.set("spot_exponent", exponent)

    def map (self, matrix: gm.Mat4) -> dict[str, Any]:
      position = self.get("light_position")
      d = self.get("spot_direction")
      direction = (matrix @ gm.vec4(d[0], d[1], d[2], 0.0))[:3]
      return {"light_position": matrix @ position,
              "spot_direction": gm.normalize(direction)}
