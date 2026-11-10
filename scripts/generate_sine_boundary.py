from decimal import Decimal
import math
from pathlib import Path

from mesh_io import read_mesh, read_rows


def baselines_from_mesh(nodes, triangles, boundaries, state):
    vertices, cells = read_mesh(nodes, triangles)
    states = read_rows(state, float)
    if len(states) != len(cells):
        raise ValueError("initial state must contain one row per triangle")
            
    adjacent = {}
    for index, tri in enumerate(cells):
        for a, b in zip(tri, (*tri[1:], tri[0])):
            edge = tuple(sorted((a, b)))
            adjacent.setdefault(edge, []).append(index)
            if len(adjacent[edge]) > 2:
                raise ValueError("mesh contains a non-manifold edge")
                
    baselines = []
    seen = set()
    midpoints = set()
    for a, b, tag in read_rows(boundaries, int):
        edge = tuple(sorted((a, b)))
        if edge in seen:
            raise ValueError("duplicate boundary edge")
            
        seen.add(edge)
        if tag not in (0, 1, 2):
            raise ValueError("boundary tags must be 0, 1, or 2")
            
        if edge not in adjacent:
            raise ValueError("boundary edge is not a triangle edge")
            
        if tag != 2:
            continue
            
        index = adjacent[edge][0]
        x = vertices[a][0] / 2 + vertices[b][0] / 2
        y = vertices[a][1] / 2 + vertices[b][1] / 2
        bed = sum(vertices[i][2] / 3 for i in cells[index])
        h, hu, hv = states[index]
        u, v = (hu / h, hv / h) if h > 0 else (0.0, 0.0)
        if not all(math.isfinite(value) for value in (x, y, bed, bed + h, u, v)):
            raise ValueError("mesh and state produced a nonfinite boundary baseline")
            
        if max(abs(x), abs(y)) > (2**63 - 1) / 1e8 / 2:
            raise ValueError("source midpoint exceeds the solver's coordinate range")
            
        key = tuple(math.copysign(math.floor(abs(value * 1e8) + 0.5), value) for value in (x, y))
        if key in midpoints:
            raise ValueError("source edges have indistinguishable midpoints")
            
        midpoints.add(key)
        baselines.append((x, y, bed, h, u, v))
        
    if not baselines:
        raise ValueError("boundaries file contains no exterior source edges (tag 2)")
        
    return sorted(baselines)


def read_baselines(path, input_format = "bed_depth"):
    if input_format not in ("bed_depth", "depth"):
        raise ValueError("input format must be bed_depth or depth")
        
    with Path(path).open(encoding="utf-8") as stream:
        edges = {}
        seen = set()
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
                
            values = [float(value) for value in line.split()]
            count = 7 if input_format == "bed_depth" else 6
            if len(values) != count or not all(math.isfinite(value) for value in values):
                raise ValueError(f"expected {count} finite values at line {line_number}")
                
            if input_format == "bed_depth":
                time, x, y, bed, depth, u, v = values
                
            else:
                time, x, y, depth, u, v = values
                bed = 0.0
                
            if depth < 0 or time < 0 or not math.isfinite(bed + depth):
                raise ValueError(f"invalid time, depth, or elevation at line {line_number}")
                
            if (x, y, time) in seen:
                raise ValueError(f"duplicate sample at line {line_number}")
                
            seen.add((x, y, time))
            if (x, y) in edges and bed != edges[x, y][1]:
                raise ValueError(f"reference bed must be constant for each edge at line {line_number}")
                
            if (x, y) not in edges or time < edges[x, y][0]:
                edges[x, y] = (time, bed, depth, u, v)
                
    if not edges:
        raise ValueError("template contains no samples")
        
    return [(x, y, *edges[x, y][1:]) for x, y in sorted(edges)]


def sample_times(end_time, interval):
    if end_time < 0:
        raise ValueError("end time must be nonnegative")
        
    if interval <= 0:
        raise ValueError("sample interval must be positive")
        
    end = Decimal(str(end_time))
    dt = Decimal(str(interval))
    count = int(end // dt)
    previous = -1.0
    for index in range(count + 1):
        time = float(index * dt)
        if time <= previous:
            raise ValueError("sample interval is too small to distinguish successive times")
            
        yield time
        previous = time
    if count * dt < end:
        if end_time <= previous:
            raise ValueError("final interval is too small to distinguish its endpoint")
            
        yield end_time


def generate_wave(output, end_time = 10.0, interval = 0.025, period = 2, amplitude = 0.1, u_amplitude = 0.3, v_amplitude = 0.0, phase = 0.0, nodes = None, triangles = None, boundaries = None, state = None):
    if not math.isfinite(period) or period <= 0:
        raise ValueError("period must be finite and positive")

    times = list(sample_times(end_time, interval))
    mesh_paths = (nodes, triangles, boundaries, state)

    if any(path is None for path in mesh_paths):
        raise ValueError("nodes, triangles, boundaries, and initial state d are all required")
        
    baselines = baselines_from_mesh(*mesh_paths)
    inputs = mesh_paths
    
    output = Path(output)
    output.parent.mkdir(parents = True, exist_ok = True)
    with output.open("w", encoding = "utf-8", newline = "\n") as stream:
        for time in times:
            wave = math.sin(2 * math.pi * time / period + phase)
            for x, y, bed, mean_depth, mean_u, mean_v in baselines:
                baseline_depth = mean_depth
                depth = max(0.0, baseline_depth + amplitude * wave)
                u = mean_u + u_amplitude * wave
                v = mean_v + v_amplitude * wave
                if depth == 0.0:
                    u = v = 0.0
                    
                if not all(math.isfinite(value) for value in (depth, u, v)):
                    raise ValueError("wave parameters produced a nonfinite boundary state")
  
                stream.write(f"{time:.17g} {x:.17g} {y:.17g} {bed:.17g} {depth:.17g} {u:.17g} {v:.17g}\n")
                
    print(f"Wrote time dependent inputs to {output}")
