import sys
sys.path.append("./scripts/")
import export_bottom as eb
import generate_sine_boundary as gsb
import write_config as wc

options = eb.Options()
options.bottom_object = "Terrain"
options.nodes = "./examples/wave.nodes"
options.triangles = "./examples/wave.triangles"
options.boundaries = "./examples/wave.boundaries" #Set the edge attribute boundary_tag in Blender: 1 for drain, 2 for source
options.init_depth = "./examples/wave.init"
options.scale = 1.
options.water_object = "Water" #Set non-zero initial depth. Can be a simple plane

eb.export_mesh(options)

end_time = 20.0
gsb.generate_wave("./examples/wave.tdinput", end_time = end_time, period = 3, amplitude = 0.05, nodes = options.nodes, triangles = options.triangles, boundaries = options.boundaries, state = options.init_depth)
wc.write_config("./examples/wave.cfg", nodes = options.nodes, triangles = options.triangles, boundaries = options.boundaries, init_depth = options.init_depth, boundary_input = "./examples/wave.tdinput", end_time = end_time, dt_max = 0.005, output_prefix = "./outputs/wave")

