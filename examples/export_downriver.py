import sys
sys.path.append("./scripts/")
import export_bottom as eb
import generate_sine_boundary as gsb
import write_config as wc

options = eb.Options()
options.bottom_object = "Terrain"
options.nodes = "./examples/downriver.nodes"
options.triangles = "./examples/downriver.triangles"
options.boundaries = "./examples/downriver.boundaries" #Set the edge attribute boundary_tag in Blender: 1 for drain, 2 for source
options.init_depth = "./examples/downriver.init"
options.scale = 1.
options.water_object = None #Set non-zero initial depth. Can be a simple plane

eb.export_mesh(options)

end_time = 20.0
wc.write_config("./examples/downriver.cfg", nodes = options.nodes, triangles = options.triangles, boundaries = options.boundaries, init_depth = options.init_depth, boundary_input = "./examples/river_outflow.tdinput", end_time = end_time, dt_max = 0.005, output_prefix = "./outputs/downriver")

