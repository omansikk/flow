import sys
sys.path.append("./scripts/")
import export_bottom as eb
import write_config as wc

options = eb.Options()
options.bottom_object = "Terrain"
options.nodes = "./examples/river.nodes"
options.triangles = "./examples/river.triangles"
options.boundaries = "./examples/river.boundaries" #Set the edge attribute boundary_tag: 1 for drain, 2 for source
options.init_depth = "./examples/river.init"
options.scale = 1.
options.water_object = None #Set non-zero initial depth. Can be a simple plane

eb.export_mesh(options)
wc.write_config("./examples/river.cfg", nodes = options.nodes, triangles = options.triangles, boundaries = options.boundaries, init_depth = options.init_depth, end_time = 20., dt_max = 0.005, output_prefix = "./outputs/river", boundary_output = "./examples/river_outflow.tdinput")
