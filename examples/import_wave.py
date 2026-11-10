import sys
sys.path.append("./scripts/")
import import_water as iw

options = iw.Options()
options.nodes = "./examples/wave.nodes"
options.triangles = "./examples/wave.triangles"
options.frame_path = "./outputs/wave_*"
options.output = "./blend/wave_animation.blend"
options.name = "WaterSurface"
options.fps = 24
options.playback_speed = 1
options.unit_scale = 1
options.surface_offset = -0.002

iw.import_water(options)

