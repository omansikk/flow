from pathlib import Path


def write_config(path, nodes, triangles, boundaries, init_depth, output_prefix = None, boundary_input = None, boundary_output = None, backend = None, end_time = None, dt_max = None, gravity = None, cfl = None, dry_tolerance = None, source_depth = None, source_velocity_x = None, source_velocity_y = None, output_every = None):
    with Path(path).open("w", encoding = "utf-8", newline= "\n") as stream:
        stream.write("nodes = " + nodes + "\n")
        stream.write("triangles = " + triangles + "\n")
        stream.write("boundaries = " + boundaries + "\n")
        stream.write("state = " + init_depth + "\n")
        if output_prefix:
            stream.write("output_prefix = " + output_prefix + "\n")
            
        if boundary_input:
            stream.write("boundary_input = " + boundary_input + "\n")
            
        if boundary_output:
            stream.write("boundary_output = " + boundary_output + "\n")
    
        if backend:
            stream.write("backend = " + backend + "\n")
    
        if end_time:
            stream.write("end_time = " + str(end_time) + "\n")
    
        if dt_max:
            stream.write("dt_max = " + str(dt_max) + "\n")
    
        if gravity:
            stream.write("gravity = " + str(gravity) + "\n")
    
        if cfl:
            stream.write("cfl = " + str(cfl) + "\n")
            
        if dry_tolerance:
            stream.write("dry_tolerance = " + str(dry_tolerance) + "\n")
            
        if source_depth:
            stream.write("source_depth = " + str(source_depth) + "\n")
            
        if source_velocity_x:
            stream.write("source_velocity_x = " + str(source_velocity_x) + "\n")
            
        if source_velocity_y:
            stream.write("source_velocity_y = " + str(source_velocity_y) + "\n")
            
        if output_every:
            stream.write("output_every = " + str(output_every) + "\n")
    
