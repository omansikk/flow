"""Read the static split geometry shared by animation and plotting tools."""
import math
from pathlib import Path


def read_rows(path, convert):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            values = line.split()
            if not values:
                continue
                
            if len(values) != 3:
                raise ValueError(f"expected three values in {path} at line {line_number}")
                
            try:
                rows.append(tuple(convert(value) for value in values))
                
            except ValueError as error:
                raise ValueError(f"invalid value in {path} at line {line_number}") from error
                
    return rows


def read_mesh(nodes_path, triangles_path):
    vertices = read_rows(nodes_path, float)
    triangles = read_rows(triangles_path, int)
    if len(vertices) < 3 or not triangles:
        raise ValueError("mesh requires at least three nodes and one triangle")
        
    if any(not math.isfinite(value) for vertex in vertices for value in vertex):
        raise ValueError("node coordinates and elevations must be finite")
        
    for tri in triangles:
        if any(i < 0 or i >= len(vertices) for i in tri):
            raise ValueError("triangle node index out of range")
            
        a, b, c = (vertices[i] for i in tri)
        if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) <= 2e-12:
            raise ValueError("mesh contains a degenerate triangle")
            
    return vertices, triangles
