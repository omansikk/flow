import argparse
import glob
import math
from numpy import loadtxt
from pathlib import Path
import sys
import bpy
from mesh_io import read_mesh, read_rows


class Options():    
    nodes = "terrain.nodes"
    triangles = "terrain.triangles"
    frame_path = "./output/terrain_*"
    output = "animation.blend"
    name = "WaterSurface"
    fps = 24
    playback_speed = 1
    unit_scale = 1
    surface_offset = -0.002


def find_frame_times(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths and Path(pattern).is_file():
        paths = [pattern]
        
    if not paths:
        raise ValueError(f"no files match {pattern!r}")
        
    frames = []
    for index, path in enumerate(paths):
        with open(path) as f:
            first_line = f.readline().lstrip("time = ")
            time = float(first_line)
            frames.append((time, path))
        
    if any(frames[i][0] >= frames[i + 1][0] for i in range(len(frames) - 1)):
        raise ValueError("simulation times must increase")
        
    return frames


def read_depths(path, triangle_count):
    depths = loadtxt(path, skiprows = 1)
            
    if len(depths) != triangle_count:
        raise ValueError(f"{path} has {len(depths)} cells; mesh has {triangle_count} triangles")
        
    return depths


def build_vertex_incidence(vertices, triangles):
    incident_cells = [[] for _ in vertices]
    for cell_index, tri in enumerate(triangles):
        a, b, c = (vertices[index] for index in tri)
        area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * 0.5
        if area <= 0:
            raise ValueError("mesh contains a degenerate triangle")
            
        for vertex_index in tri:
            incident_cells[vertex_index].append((cell_index, area))
            
    return incident_cells


def surface_vertices(vertices, incident_cells, depths, unit_scale, surface_offset):
    nodal_depth = make_vertex_depths(incident_cells, depths)
    
    return [(x * unit_scale, y * unit_scale, (z + depth) * unit_scale + surface_offset) for (x, y, z), depth in zip(vertices, nodal_depth)]


def make_vertex_depths(incident_cells, depths):
    nodal_depths = []
    for cells in incident_cells:
        weighted_sum = 0.0
        total_area = 0.0
        for cell_index, area in cells:
            weighted_sum += max(0.0, depths[cell_index]) * area
            total_area += area
            
        nodal_depths.append(weighted_sum / total_area if total_area else 0.0)
        
    return nodal_depths


def animation_timeline(samples, fps, playback_speed):
    if not samples:
        raise ValueError("animation requires at least one sample")
        
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(playback_speed) or playback_speed <= 0:
        raise ValueError("fps and playback speed must be finite and positive")
        
    t0 = samples[0][0]
    return [1 + (time - t0) * fps / playback_speed for time, _ in samples]


def animate_absolute_keys(shape_keys, timeline, key_times, edit_preferences):
    if len(timeline) != len(key_times) or not timeline:
        raise ValueError("each animation frame must have one shape-key time")
        
    previous_interpolation = edit_preferences.keyframe_new_interpolation_type
    try:
        edit_preferences.keyframe_new_interpolation_type = "LINEAR"
        for frame, key_time in zip(timeline, key_times):
            shape_keys.eval_time = key_time
            shape_keys.keyframe_insert(data_path="eval_time", frame=frame, group="Simulation time")
            
    finally:
        edit_preferences.keyframe_new_interpolation_type = previous_interpolation


def create_bottom_mesh(vertices, triangles, name, unit_scale):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata([tuple(value * unit_scale for value in vertex) for vertex in vertices], [], triangles)
    mesh.update()
    bottom = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(bottom)
    
    return bottom


def import_water(options):
    vertices, triangles = read_mesh(options.nodes, options.triangles)
    samples = find_frame_times(options.frame_path)
    vertex_incidence = build_vertex_incidence(vertices, triangles)
    timeline = animation_timeline(samples, options.fps, options.playback_speed)

    first_time, first_path = samples[0]
    first_surface = surface_vertices( vertices, vertex_incidence, read_depths(first_path, len(triangles)), options.unit_scale, options.surface_offset)

    bottom = create_bottom_mesh(vertices, triangles, "Bottom", options.unit_scale)

    mesh = bpy.data.meshes.new(options.name + "Mesh")
    mesh.from_pydata(first_surface, [], triangles)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
        
    obj = bpy.data.objects.new(options.name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Absolute shape keys keep a single mesh topology and interpolate between solver samples.
    basis = obj.shape_key_add(name = "Basis", from_mix = False)
    shape_keys = obj.data.shape_keys
    shape_keys.use_relative = False
    key_times = [basis.frame]
    if hasattr(basis, "interpolation"):
        basis.interpolation = "KEY_LINEAR"
        
    for index, (time, path) in enumerate(samples[1:], start = 1):
        key = obj.shape_key_add(name = f"Water_{index:05d}", from_mix = False)
        key_times.append(key.frame)
        if hasattr(key, "interpolation"):
            key.interpolation = "KEY_LINEAR"
            
        depths = read_depths(path, len(triangles))
        surface = surface_vertices(vertices, vertex_incidence, depths, options.unit_scale, options.surface_offset)
            
        for point, co in zip(key.data, surface):
            point.co = co

    material = bpy.data.materials.new(options.name + "Material")
    material.diffuse_color = (0.025, 0.24, 0.62, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = material.diffuse_color
        principled.inputs["Roughness"].default_value = 0.1
        
    mesh.materials.append(material)

    animate_absolute_keys(shape_keys, timeline, key_times, bpy.context.preferences.edit)

    scene = bpy.context.scene
    scene.render.fps = round(options.fps)
    scene.frame_start = math.floor(timeline[0])
    scene.frame_end = max(scene.frame_start, math.ceil(timeline[-1]))
    scene.frame_set(scene.frame_start)
    
    output = Path(options.output).expanduser().resolve()
    output.parent.mkdir(parents = True, exist_ok = True)
    bpy.ops.wm.save_as_mainfile(filepath = str(output))


if __name__ == "__main__":
    options = Options()
    import_water(options)
