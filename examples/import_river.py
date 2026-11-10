import sys
sys.path.append("./scripts/")
import import_water as iw

options = iw.Options()
options.nodes = "./examples/river.nodes"
options.triangles = "./examples/river.triangles"
options.frame_path = "./outputs/river_*"
options.output = "./blend/river_animation.blend"
options.name = "WaterSurface"
options.fps = 24
options.playback_speed = 1
options.unit_scale = 1
options.surface_offset = -0.002

iw.import_water(options)

