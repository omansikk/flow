import sys
sys.path.append("./scripts/")
import import_water as iw

options = iw.Options()
options.nodes = "./examples/downriver.nodes"
options.triangles = "./examples/downriver.triangles"
options.frame_path = "./outputs/downriver_*"
options.output = "./blend/downriver_animation.blend"
options.name = "WaterSurface"
options.fps = 24
options.playback_speed = 1
options.unit_scale = 1
options.surface_offset = -0.002

iw.import_water(options)

