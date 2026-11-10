import math
from collections import Counter
from pathlib import Path


class Options:
    bottom_object = "Terrain"
    nodes = "terrain.nodes"
    triangles = "terrain.triangles"
    boundaries = "terrain.boundaries"
    init_depth = "terrain.init"
    scale = 1.
    water_object = None


def initial_depths(vertices, triangles, surface_height=None):
    depths = []
    for triangle in triangles:
        x, y, bed = (sum(vertices[i][axis] / 3.0 for i in triangle)
                     for axis in range(3))
        elevation = surface_height(x, y) if surface_height is not None else None
        if elevation is None:
            depths.append(0.0)
        elif not math.isfinite(elevation):
            raise ValueError("water surface elevation must be finite")
            
        else:
            depth = max(0.0, elevation - bed)
            if not math.isfinite(depth):
                raise ValueError("initial depth must be finite")
            depths.append(depth)
            
    return depths


def water_surface_sampler(obj, depsgraph, scale):
    from mathutils.bvhtree import BVHTree

    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertices = []
        for vertex in mesh.vertices:
            world = evaluated.matrix_world @ vertex.co
            point = tuple(value * scale for value in (world.x, world.y, world.z))
            if not all(math.isfinite(value) for value in point):
                raise ValueError("water mesh coordinates must be finite")
                
            vertices.append(point)
            
        triangles = [tuple(tri.vertices) for tri in mesh.loop_triangles]
        if not vertices or not triangles:
            raise ValueError("water mesh has no vertices or faces")
            
        tree = BVHTree.FromPolygons(vertices, triangles, all_triangles = True)
        top = max(point[2] for point in vertices)
        ray_z = top + max(1.0, abs(top) * 1e-6)
        if not math.isfinite(ray_z):
            raise ValueError("water mesh elevation is too large")
            
    finally:
        evaluated.to_mesh_clear()

    def height(x, y):
        location, _normal, _index, _distance = tree.ray_cast((x, y, ray_z), (0.0, 0.0, -1.0))
            
        return None if location is None else location.z

    return height


def write_state(path, depths):
    with path.open("w", encoding = "utf-8", newline = "\n") as stream:
        for depth in depths:
            stream.write(f"{depth:.12g} 0 0\n")


def export_mesh(options):
    import bpy
    
    obj = bpy.data.objects.get(options.bottom_object)
    if obj is None:
        raise SystemExit(f"object not found: {options.bottom_object}")

    if obj.type != "MESH":
        raise SystemExit(f"object {options.bottom_object!r} is not a mesh")

    if not math.isfinite(options.scale) or options.scale <= 0:
        raise SystemExit("scale must be finite and positive")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    surface_height = None
    if options.water_object:
        water_obj = bpy.data.objects.get(options.water_object)
        if water_obj is None or water_obj.type != "MESH":
            raise SystemExit(f"water mesh object not found: {options.water_object}")
            
        if water_obj == obj:
            raise SystemExit("water mesh must be a different mesh from bottom")
        surface_height = water_surface_sampler(water_obj, depsgraph, options.scale)

    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    
    try:
        mesh.calc_loop_triangles()
        if not mesh.vertices or not mesh.loop_triangles:
            raise SystemExit("mesh has no vertices or faces")

        vertices = []
        for vertex in mesh.vertices:
            world = evaluated.matrix_world @ vertex.co
            point = (world.x * options.scale, world.y * options.scale, world.z * options.scale)
            vertices.append(point)

        triangles = [tuple(int(i) for i in tri.vertices) for tri in mesh.loop_triangles]
        depths = initial_depths(vertices, triangles, surface_height)
        edge_use = Counter()
        for a, b, c in triangles:
            for i, j in ((a, b), (b, c), (c, a)):
                edge_use[tuple(sorted((i, j)))] += 1

        edge_attr = mesh.attributes.get("boundary_tag")
        tagged_edges = {}
        if edge_attr is not None:
            for edge, item in zip(mesh.edges, edge_attr.data):
                tag = int(item.value)
                if tag not in (0, 1, 2):
                    raise SystemExit(f"unsupported boundary tag; use 0, 1, or 2")

                pair = tuple(sorted((int(edge.vertices[0]), int(edge.vertices[1]))))
                if tag:
                    tagged_edges[pair] = tag

        boundary_edges = {}
        for edge, count in edge_use.items():
            if count > 2:
                raise SystemExit(f"non-manifold edge {edge} is shared by {count} triangles")

            if count == 1 or edge in tagged_edges:
                boundary_edges[edge] = tagged_edges.get(edge, 0)

        state_path = options.init_depth or str(Path(options.nodes).with_suffix(".init"))
        outputs = [Path(path).expanduser().resolve() for path in (options.nodes, options.triangles, options.boundaries, state_path)]
                   
        for output in outputs:
            output.parent.mkdir(parents = True, exist_ok = True)
            
        with outputs[0].open("w", encoding = "utf-8", newline = "\n") as stream:
            for x, y, z in vertices:
                stream.write(f"{x:.12g} {y:.12g} {z:.12g}\n")

        with outputs[1].open("w", encoding = "utf-8", newline = "\n") as stream:
            for a, b, c in triangles:
                stream.write(f"{a} {b} {c}\n")

        with outputs[2].open("w", encoding = "utf-8", newline = "\n") as stream:
            for (a, b), tag in sorted(boundary_edges.items()):
                stream.write(f"{a} {b} {tag}\n")

        write_state(outputs[3], depths)

        print("Exported " + f"{options.nodes}, {options.triangles}, {options.boundaries}, and {options.init_depth}")
        
    finally:
        evaluated.to_mesh_clear()


if __name__ == "__main__":
    options = Options()
    options.bottom_object = "Terrain"
    options.nodes = "terrain.nodes"
    options.triangles = "terrain.triangles"
    options.boundaries = "terrain.boundaries" #Set the edge attribute boundary_tag: 1 for drain, 2 for source
    options.init_depth = "terrain.init"
    options.scale = 1.
    options.water_object = None #Set non-zero initial depth. Can be a simple plane
    
    export_mesh(options)
