#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef CUDA
#include "cuda_backend.h"
#endif

namespace {

constexpr double num_zero = 1e-12;
constexpr double boundary_scale = 1e8;


struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};


struct Node {
    Vec2 position;
    double bed = 0.0;
};


struct Cell {
    std::array<int, 3> nodes{};
    double area = 0.0;
    Vec2 center;
    double bed = 0.0;
    double bed_dx = 0.0;
    double bed_dy = 0.0;
    double depth = 0.0;
    double depth_x_momentum = 0.0;
    double depth_y_momentum = 0.0;
};


struct Face {
    int left_cell = -1;
    int right_cell = -1;
    int boundary_tag = 0;
    Vec2 normal;
    Vec2 midpoint;
    double length = 0.0;
};


struct PrimitiveState {
    double depth = 0.0;
    double velocity_x = 0.0;
    double velocity_y = 0.0;
};


struct BoundaryState {
    double depth = 0.0;
    double velocity_x = 0.0;
    double velocity_y = 0.0;
    double bed = 0.0;
    bool has_bed = false;
};


struct BoundarySample {
    double time = 0.0;
    BoundaryState state;
};


struct Flux {
    double depth = 0.0;
    double x_momentum = 0.0;
    double y_momentum = 0.0;
};


struct Mesh {
    std::vector<Node> nodes;
    std::vector<Cell> cells;
    std::vector<Face> faces;
};


using EdgeKey = std::pair<int, int>;
using PointKey = std::pair<long long, long long>;
using BoundarySeries = std::map<PointKey, std::vector<BoundarySample>>;
using Residual = std::array<double, 3>;


EdgeKey edge_key(int a, int b) {
    return {std::min(a, b), std::max(a, b)};
}


PointKey point_key(Vec2 point) {
    return {
        std::llround(point.x * boundary_scale),
        std::llround(point.y * boundary_scale),
    };
}


double cross(Vec2 a, Vec2 b, Vec2 c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}


Vec2 midpoint(Vec2 a, Vec2 b) {
    return {0.5 * (a.x + b.x), 0.5 * (a.y + b.y)};
}


Vec2 outward_normal(Vec2 a, Vec2 b, Vec2 cell_center) {
    Vec2 edge{b.x - a.x, b.y - a.y};
    const double length = std::hypot(edge.x, edge.y);
    if (length <= num_zero) {
        throw std::runtime_error("mesh contains a zero-length edge");
    }

    Vec2 normal{edge.y / length, -edge.x / length};
    const Vec2 face_center = midpoint(a, b);
    const Vec2 toward_face{face_center.x - cell_center.x, face_center.y - cell_center.y};
    if (normal.x * toward_face.x + normal.y * toward_face.y < 0.0) {
        normal.x = -normal.x;
        normal.y = -normal.y;
    }
    
    return normal;
}


template <typename T, std::size_t Columns = 3>
std::vector<std::array<T, Columns>> read_rows(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open input: " + path);
    }
    
    std::vector<std::array<T, Columns>> rows;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        std::istringstream record(line);
        record >> std::ws;
        if (record.eof()) {
            continue;
        }
        
        std::array<T, Columns> row{};
        for (T& value : row) {
            if (!(record >> value)) {
                throw std::runtime_error("expected " + std::to_string(Columns)  + " numeric values in " + path + " at line " + std::to_string(line_number));
            }
        }
        
        if (!(record >> std::ws).eof()) {
            throw std::runtime_error("expected " + std::to_string(Columns) + " values in " + path + " at line " + std::to_string(line_number));
        }
        
        rows.push_back(row);
    }
    
    if (input.bad()) {
        throw std::runtime_error("error reading data in " + path);
    }
    
    return rows;
}


Mesh read_mesh(const std::string& nodes_path, const std::string& triangles_path, const std::string& boundaries_path, const std::string& state_path) {
    const auto nodes = read_rows<double>(nodes_path);
    const std::size_t node_count = nodes.size();
    if (node_count < 3) {
        throw std::runtime_error("mesh requires at least three nodes");
    }

    Mesh mesh;
    mesh.nodes.reserve(node_count);
    for (const auto& row : nodes) {
        if (!std::isfinite(row[0]) || !std::isfinite(row[1]) || !std::isfinite(row[2])) {
            throw std::runtime_error("incomplete or invalid node data in " + nodes_path);
        }
        
        mesh.nodes.push_back({{row[0], row[1]}, row[2]});
    }

    const auto triangles = read_rows<int>(triangles_path);
    const std::size_t cell_count = triangles.size();
    if (cell_count == 0) {
        throw std::runtime_error("mesh requires at least one triangle");
    }

    mesh.cells.resize(cell_count);
    for (std::size_t index = 0; index < cell_count; ++index) {
        Cell& cell = mesh.cells[index];
        cell.nodes = triangles[index];
        for (int node_index : cell.nodes) {
            if (node_index < 0 || static_cast<std::size_t>(node_index) >= node_count) {
                throw std::runtime_error("triangle node index out of range");
            }
        }

        const Node& a = mesh.nodes[cell.nodes[0]];
        const Node& b = mesh.nodes[cell.nodes[1]];
        const Node& c = mesh.nodes[cell.nodes[2]];
        const double signed_area_twice = cross(a.position, b.position, c.position);
        cell.area = 0.5 * std::abs(signed_area_twice);
        if (cell.area <= num_zero) {
            throw std::runtime_error("mesh contains a degenerate triangle");
        }

        cell.center = {
            (a.position.x + b.position.x + c.position.x) / 3.0,
            (a.position.y + b.position.y + c.position.y) / 3.0,
        };
        
        cell.bed = (a.bed + b.bed + c.bed) / 3.0;
        cell.bed_dx = ((b.bed - a.bed) * (c.position.y - a.position.y) - (c.bed - a.bed) * (b.position.y - a.position.y)) / signed_area_twice;
        cell.bed_dy = ((b.position.x - a.position.x) * (c.bed - a.bed) - (c.position.x - a.position.x) * (b.bed - a.bed)) / signed_area_twice;
    }
    
    const auto boundaries = read_rows<int>(boundaries_path);

    std::map<EdgeKey, int> boundary_tags;
    for (const auto& row : boundaries) {
        const int a = row[0];
        const int b = row[1];
        const int tag = row[2];
        if (a < 0 || b < 0 || static_cast<std::size_t>(a) >= node_count
            || static_cast<std::size_t>(b) >= node_count) {
            throw std::runtime_error("boundary node index out of range");
        }
        
        if (tag < 0 || tag > 2) {
            throw std::runtime_error("boundary tag must be 0, 1, or 2");
        }
        
        if (!boundary_tags.emplace(edge_key(a, b), tag).second) {
            throw std::runtime_error("duplicate boundary edge record");
        }
    }
    
    std::map<EdgeKey, int> open_edges;
    std::set<EdgeKey> completed_edges;
    std::set<EdgeKey> used_boundary_tags;

    for (std::size_t cell_index = 0; cell_index < mesh.cells.size(); ++cell_index) {
        const Cell& cell = mesh.cells[cell_index];
        for (int side = 0; side < 3; ++side) {
            const int a_index = cell.nodes[side];
            const int b_index = cell.nodes[(side + 1) % 3];
            const EdgeKey key = edge_key(a_index, b_index);
            if (completed_edges.count(key)) {
                throw std::runtime_error("mesh has a non-manifold edge shared by more than two cells");
            }

            const auto owner = open_edges.find(key);
            if (owner == open_edges.end()) {
                open_edges.emplace(key, static_cast<int>(cell_index));
                continue;
            }

            const int left_index = owner->second;
            open_edges.erase(owner);
            completed_edges.insert(key);

            const Vec2 a = mesh.nodes[a_index].position;
            const Vec2 b = mesh.nodes[b_index].position;
            const Vec2 center_delta{
                cell.center.x - mesh.cells[left_index].center.x,
                cell.center.y - mesh.cells[left_index].center.y,
            };
            
            Vec2 normal = outward_normal(a, b, mesh.cells[left_index].center);
            if (normal.x * center_delta.x + normal.y * center_delta.y < 0.0) {
                normal.x = -normal.x;
                normal.y = -normal.y;
            }

            const auto tag = boundary_tags.find(key);
            if (tag != boundary_tags.end()) {
                used_boundary_tags.insert(key);
            }
            
            mesh.faces.push_back({
                left_index,
                static_cast<int>(cell_index),
                tag == boundary_tags.end() ? 0 : tag->second,
                normal,
                midpoint(a, b),
                std::hypot(b.x - a.x, b.y - a.y),
            });
        }
    }

    for (const auto& [key, cell_index] : open_edges) {
        const Vec2 a = mesh.nodes[key.first].position;
        const Vec2 b = mesh.nodes[key.second].position;
        const auto tag = boundary_tags.find(key);
        if (tag != boundary_tags.end()) {
            used_boundary_tags.insert(key);
        }
        
        mesh.faces.push_back({
            cell_index,
            -1,
            tag == boundary_tags.end() ? 0 : tag->second,
            outward_normal(a, b, mesh.cells[cell_index].center),
            midpoint(a, b),
            std::hypot(b.x - a.x, b.y - a.y),
        });
    }

    if (used_boundary_tags.size() != boundary_tags.size()) {
        throw std::runtime_error("a tagged boundary edge is not part of the mesh");
    }
    
    const auto state = read_rows<double>(state_path);
    if (state.size() != mesh.cells.size()) {
        throw std::runtime_error("state cell count does not match mesh");
    }
    
    for (std::size_t index = 0; index < state.size(); ++index) {
        Cell& cell = mesh.cells[index];
        cell.depth = state[index][0];
        cell.depth_x_momentum = state[index][1];
        cell.depth_y_momentum = state[index][2];
        if (!std::isfinite(cell.depth) || cell.depth < 0.0
                      || !std::isfinite(cell.depth_x_momentum)
                      || !std::isfinite(cell.depth_y_momentum)) {
            throw std::runtime_error("state depth and momentum must be finite; depth must be nonnegative");
        }
    }
    
    return mesh;
}


BoundarySeries read_boundary_series(const std::string& path, bool bed_depth = true) {
    BoundarySeries series;
    const auto add_rows = [&](const auto& rows) {
        for (const auto& row : rows) {
            if (!std::all_of(row.begin(), row.end(), [](double value) { return std::isfinite(value); })) {
                throw std::runtime_error("boundary samples must be finite");
            }
            
            const std::size_t h = bed_depth ? 4 : 3;
            BoundarySample sample{row[0], {row[h], row[h + 1], row[h + 2], bed_depth ? row[3] : 0.0, bed_depth}};
            const double coordinate_limit = static_cast<double>(std::numeric_limits<long long>::max()) / boundary_scale / 2.0;
            if (sample.time < 0.0 || sample.state.depth < 0.0
                || !std::isfinite(sample.state.bed + sample.state.depth)
                || std::abs(row[1]) > coordinate_limit || std::abs(row[2]) > coordinate_limit) {
                throw std::runtime_error("invalid boundary time, depth, elevation, or midpoint");
            }
            
            series[point_key({row[1], row[2]})].push_back(sample);
        }
    };
    
    if (bed_depth) {
        add_rows(read_rows<double, 7>(path));
    } 
    
    else {
        add_rows(read_rows<double, 6>(path));
    }
    
    if (series.empty()) {
        throw std::runtime_error("boundary series contains no samples");
    }

    for (auto& [point, samples] : series) {
        std::sort(samples.begin(), samples.end(), [](const BoundarySample& a, const BoundarySample& b) {
            return a.time < b.time;
        });
        
        for (std::size_t i = 1; i < samples.size(); ++i) {
            if (samples[i].time <= samples[i - 1].time) {
                throw std::runtime_error("boundary sample times must increase for each edge");
            }
            
            if (samples[i].state.bed != samples.front().state.bed) {
                throw std::runtime_error("boundary reference bed must be constant for each edge");
            }
        }
    }
    
    return series;
}

BoundaryState sample_boundary(const BoundarySeries& series, Vec2 edge_midpoint, double time) {
    const auto found = series.find(point_key(edge_midpoint));
    if (found == series.end()) {
        throw std::runtime_error( "no boundary series for source-edge midpoint " + std::to_string(edge_midpoint.x) + "," + std::to_string(edge_midpoint.y));
    }

    const auto& samples = found->second;
    if (time <= samples.front().time) {
        return samples.front().state;
    }
    
    if (time >= samples.back().time) {
        return samples.back().state;
    }

    const auto upper = std::upper_bound(samples.begin(), samples.end(), time, [](double query_time, const BoundarySample& sample) { 
        return query_time < sample.time;
    });
    
    const BoundarySample& before = *(upper - 1);
    const BoundarySample& after = *upper;
    const double alpha = (time - before.time) / (after.time - before.time);

 
    return {
        (1.0 - alpha) * before.state.depth + alpha * after.state.depth,
        before.state.velocity_x + alpha * (after.state.velocity_x - before.state.velocity_x),
        before.state.velocity_y + alpha * (after.state.velocity_y - before.state.velocity_y),
        before.state.bed, before.state.has_bed,
    };
}

PrimitiveState to_primitive(const Cell& cell, double dry_tolerance) {
    const double depth = std::max(cell.depth, 0.0);
    if (depth <= dry_tolerance) {
        return {depth, 0.0, 0.0};
    }
    
    return {
        depth,
        cell.depth_x_momentum / depth,
        cell.depth_y_momentum / depth,
    };
}

Flux physical_flux(PrimitiveState state, Vec2 normal, double gravity) {
    const double normal_velocity = state.velocity_x * normal.x + state.velocity_y * normal.y;
    return {
        state.depth * normal_velocity,
        state.depth * state.velocity_x * normal_velocity + 0.5 * gravity * state.depth * state.depth * normal.x,
        state.depth * state.velocity_y * normal_velocity + 0.5 * gravity * state.depth * state.depth * normal.y,
    };
}

Flux hll_flux(PrimitiveState left, PrimitiveState right, Vec2 normal, double gravity) {
    const Flux left_flux = physical_flux(left, normal, gravity);
    const Flux right_flux = physical_flux(right, normal, gravity);
    const double left_normal_velocity = left.velocity_x * normal.x + left.velocity_y * normal.y;
    const double right_normal_velocity = right.velocity_x * normal.x + right.velocity_y * normal.y;
    const double left_wave = left_normal_velocity - std::sqrt(gravity * left.depth);
    const double right_wave = right_normal_velocity + std::sqrt(gravity * right.depth);
    const double s_left = std::min(left_wave, right_normal_velocity - std::sqrt(gravity * right.depth));
    const double s_right = std::max(left_normal_velocity + std::sqrt(gravity * left.depth), right_wave);

    if (s_left >= 0.0) {
        return left_flux;
    }
    
    if (s_right <= 0.0) {
        return right_flux;
    }

    const double inverse_wave_span = 1.0 / (s_right - s_left);
    
    return {
        (s_right * left_flux.depth - s_left * right_flux.depth  + s_left * s_right * (right.depth - left.depth)) * inverse_wave_span,
        (s_right * left_flux.x_momentum - s_left * right_flux.x_momentum  + s_left * s_right * (right.depth * right.velocity_x - left.depth * left.velocity_x)) * inverse_wave_span,
        (s_right * left_flux.y_momentum - s_left * right_flux.y_momentum + s_left * s_right * (right.depth * right.velocity_y - left.depth * left.velocity_y)) * inverse_wave_span,
    };
}

PrimitiveState boundary_ghost_state(const Mesh& mesh, const Face& face, PrimitiveState left, BoundaryState source_state, double dry_tolerance) {
    if (face.boundary_tag == 2 && face.right_cell < 0) {
        if (source_state.depth <= dry_tolerance) {
            return {source_state.depth, 0.0, 0.0};
        }
        
        return {source_state.depth, source_state.velocity_x, source_state.velocity_y};
    }
    
    if (face.right_cell >= 0 && face.boundary_tag != 1) {
        return to_primitive(mesh.cells[face.right_cell], dry_tolerance);
    }
    
    if (face.boundary_tag == 1) {
        return {};
    }

    const double normal_velocity = left.velocity_x * face.normal.x + left.velocity_y * face.normal.y;
    return {
        left.depth,
        left.velocity_x - 2.0 * normal_velocity * face.normal.x,
        left.velocity_y - 2.0 * normal_velocity * face.normal.y,
    };
}

double ghost_bed(const Mesh& mesh, const Face& face, const BoundaryState& source) {
    if (face.boundary_tag == 2 && face.right_cell < 0 && source.has_bed) {
        return source.bed;
    }
    
    return face.right_cell >= 0 && face.boundary_tag != 1 ? mesh.cells[face.right_cell].bed : mesh.cells[face.left_cell].bed;
}

PrimitiveState reconstruct(PrimitiveState state, double bed, double crest, double dry_tolerance) {
    state.depth = std::max(0.0, state.depth - (crest - bed));
    if (state.depth <= dry_tolerance) {
        state.velocity_x = state.velocity_y = 0.0;
    }
    
    return state;
}

void compute_residuals(const Mesh& mesh, double gravity, double dry_tolerance, const std::vector<BoundaryState>& boundary_states, std::vector<Residual>& residuals) {
    std::fill(residuals.begin(), residuals.end(), Residual{0.0, 0.0, 0.0});

#ifdef OPENMP
#pragma omp parallel for
#endif
    for (std::ptrdiff_t face_index = 0;
         face_index < static_cast<std::ptrdiff_t>(mesh.faces.size()); ++face_index) {
        const Face& face = mesh.faces[face_index];
        const Cell& left_cell = mesh.cells[face.left_cell];
        const PrimitiveState left = to_primitive(left_cell, dry_tolerance);
        const PrimitiveState right = boundary_ghost_state( mesh, face, left, boundary_states[face_index], dry_tolerance);

        const double right_bed = ghost_bed(mesh, face, boundary_states[face_index]);
        const double reconstructed_bed = std::max(left_cell.bed, right_bed);
        const PrimitiveState left_face = reconstruct(left, left_cell.bed, reconstructed_bed, dry_tolerance);
        const PrimitiveState right_face = reconstruct(right, right_bed, reconstructed_bed, dry_tolerance);
        
        const double left_depth = left_face.depth;
        const double right_depth = right_face.depth;
        const Flux flux = hll_flux(left_face, right_face, face.normal, gravity);

        const double left_pressure_correction = 0.5 * gravity * (left.depth * left.depth - left_depth * left_depth);
        const double left_terms[3] = {
            -flux.depth * face.length,
            -(flux.x_momentum + left_pressure_correction * face.normal.x) * face.length,
            -(flux.y_momentum + left_pressure_correction * face.normal.y) * face.length,
        };
        
#ifdef OPENMP
#pragma omp atomic update
#endif
        residuals[face.left_cell][0] += left_terms[0];
        
#ifdef OPENMP
#pragma omp atomic update
#endif
        residuals[face.left_cell][1] += left_terms[1];
        
#ifdef OPENMP
#pragma omp atomic update
#endif
        residuals[face.left_cell][2] += left_terms[2];

        if (face.right_cell >= 0 && face.boundary_tag != 1) {
            const double right_pressure_correction = 0.5 * gravity * (right.depth * right.depth - right_depth * right_depth);
            const double right_terms[3] = {
                flux.depth * face.length,
                (flux.x_momentum + right_pressure_correction * face.normal.x) * face.length,
                (flux.y_momentum + right_pressure_correction * face.normal.y) * face.length,
            };
            
#ifdef OPENMP
#pragma omp atomic update
#endif
            residuals[face.right_cell][0] += right_terms[0];
            
#ifdef OPENMP
#pragma omp atomic update
#endif
            residuals[face.right_cell][1] += right_terms[1];
            
#ifdef OPENMP
#pragma omp atomic update
#endif
            residuals[face.right_cell][2] += right_terms[2];
        }
    }
}

void update_cells(Mesh& mesh, const std::vector<Residual>& residuals, double dt, double dry_tolerance) {
#ifdef OPENMP
#pragma omp parallel for
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(mesh.cells.size()); ++i) {
        Cell& cell = mesh.cells[i];
        cell.depth = std::max(0.0, cell.depth + dt * residuals[i][0] / cell.area);
        cell.depth_x_momentum += dt * residuals[i][1] / cell.area;
        cell.depth_y_momentum += dt * residuals[i][2] / cell.area;
        if (cell.depth <= dry_tolerance) {
            cell.depth = 0.0;
            cell.depth_x_momentum = 0.0;
            cell.depth_y_momentum = 0.0;
        }
    }
}

double choose_time_step(const Mesh& mesh, double time_remaining, double dt_max, double gravity, double cfl, double dry_tolerance, const std::vector<BoundaryState>& boundary_states) {
    double dt = std::min(dt_max, time_remaining);
    std::vector<double> spectral_sum(mesh.cells.size(), 0.0);
    for (std::size_t i = 0; i < mesh.faces.size(); ++i) {
        const Face& face = mesh.faces[i];
        const Cell& cell = mesh.cells[face.left_cell];
        const double bed = ghost_bed(mesh, face, boundary_states[i]);
        const double crest = std::max(cell.bed, bed);
        const PrimitiveState left = reconstruct(to_primitive(cell, dry_tolerance), cell.bed, crest, dry_tolerance);
        const PrimitiveState right = reconstruct(boundary_ghost_state(mesh, face, to_primitive(cell, dry_tolerance), boundary_states[i], dry_tolerance), bed, crest, dry_tolerance);
        const auto speed = [&](PrimitiveState state) {
            return std::abs(state.velocity_x * face.normal.x + state.velocity_y * face.normal.y) + std::sqrt(gravity * state.depth);
        };
        
        const double contribution = face.length * std::max(speed(left), speed(right));
        spectral_sum[face.left_cell] += contribution;
        if (face.right_cell >= 0 && face.boundary_tag != 1) {
            spectral_sum[face.right_cell] += contribution;
        }
    }
    
    for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
        if (spectral_sum[i] > num_zero) {
            dt = std::min(dt, cfl * mesh.cells[i].area / spectral_sum[i]);
        }
    }
    
    if (!(dt > 0.0) || !std::isfinite(dt)) {
        throw std::runtime_error("no finite positive timestep; check mesh and boundary states");
    }
    
    return dt;
}


void write(const std::string& path, const Mesh& mesh, double time) {
    const std::filesystem::path output_path(path);
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("cannot write output: " + path);
    }

    output << "time = " << time << "\n" << std::setprecision(10);
    for (const Cell& cell : mesh.cells) {
        output << cell.depth << '\n';
    }
}


void write_boundary_snapshot(std::ostream& output, const Mesh& mesh, double time, double dry_tolerance) {
    output << std::setprecision(17);
    for (const Face& face : mesh.faces) {
        if (face.boundary_tag != 1) {
            continue;
        }
        
        const Cell& cell = mesh.cells[face.left_cell];
        const PrimitiveState state = to_primitive(cell, dry_tolerance);
        output << time << ' ' << face.midpoint.x << ' ' << face.midpoint.y << ' ' << cell.bed << ' ' << state.depth << ' '<< state.velocity_x << ' ' << state.velocity_y << '\n';
    }
    
    output.flush();
    if (!output) {
        throw std::runtime_error("failed writing boundary snapshot");
    }
}


struct Options {
    std::string nodes_path;
    std::string triangles_path;
    std::string boundaries_path;
    std::string state_path;
    std::string output_prefix = "frame";
    std::string boundary_input_path;
    std::string boundary_output_path;
    std::string boundary_input_format = "bed_depth";
    std::string backend = "cpu";
    double end_time = 0.5;
    double dt_max = 0.01;
    double gravity = 9.81;
    double cfl = 0.5;
    double dry_tolerance = 1e-6;
    double source_depth = 0.35;
    double source_velocity_x = 1.0;
    double source_velocity_y = 0.0;
    int output_every = 10;
};


std::string trim(std::string value) {
    const auto is_not_space = [](unsigned char character) {
        return !std::isspace(character);
    };
    
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), is_not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), is_not_space).base(), value.end());
    
    return value;
}


std::filesystem::path resolve_config_path(const std::filesystem::path& base_directory, const std::string& value) {
    const std::filesystem::path path(value);
    
    return path.is_absolute() ? path : (base_directory / path).lexically_normal();
}


Options read_options(const std::string& config_path) {
    std::ifstream input(config_path);
    if (!input) {
        throw std::runtime_error("cannot open run input file: " + config_path);
    }

    const std::filesystem::path absolute_config = std::filesystem::absolute(config_path);
    const std::filesystem::path base_directory = absolute_config.parent_path().parent_path();
    Options options;
    std::map<std::string, std::string> values;
    std::string line;
    std::size_t line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const std::size_t equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error("expected key = value at line " + std::to_string(line_number));
        }
        
        const std::string key = trim(line.substr(0, equals));
        const std::string value = trim(line.substr(equals + 1));
        if (key.empty() || value.empty()) {
            throw std::runtime_error("empty key or value at line " + std::to_string(line_number));
        }
        
        if (!values.emplace(key, value).second) {
            throw std::runtime_error("duplicate setting '" + key + "'");
        }
    }
    
    const auto take = [&](const std::string& key, const std::string& fallback) {
        const auto found = values.find(key);
        if (found == values.end()) {
            return fallback;
        }
        
        const std::string value = found->second;
        values.erase(found);
        
        return value;
    };
    
    const auto take_required = [&](const std::string& key) {
        const std::string value = take(key, "");
        if (value.empty()) {
            throw std::runtime_error("missing required setting '" + key + "'");
        }
        
        return value;
    };

    options.nodes_path = resolve_config_path(base_directory, take_required("nodes")).string();
    options.triangles_path = resolve_config_path(base_directory, take_required("triangles")).string();
    options.boundaries_path = resolve_config_path(base_directory, take_required("boundaries")).string();
    options.state_path = resolve_config_path(base_directory, take_required("state")).string();
    options.output_prefix = resolve_config_path(base_directory, take("output_prefix", options.output_prefix)).string();
    options.boundary_input_path = take("boundary_input", "");
    options.boundary_output_path = take("boundary_output", "");
    options.boundary_input_format = take("boundary_input_format", options.boundary_input_format);
    if (options.boundary_input_format != "bed_depth" && options.boundary_input_format != "depth") {
        throw std::runtime_error("boundary_input_format must be bed_depth (7 columns) or depth (6 columns)");
    }
    
    if (!options.boundary_input_path.empty()) {
        options.boundary_input_path = resolve_config_path(base_directory, options.boundary_input_path).string();
    }
    
    if (!options.boundary_output_path.empty()) {
        options.boundary_output_path = resolve_config_path(base_directory, options.boundary_output_path).string();
    }

    options.backend = take("backend", options.backend);
    options.end_time = std::stod(take("end_time", std::to_string(options.end_time)));
    options.dt_max = std::stod(take("dt_max", std::to_string(options.dt_max)));
    options.gravity = std::stod(take("gravity", std::to_string(options.gravity)));
    options.cfl = std::stod(take("cfl", std::to_string(options.cfl)));
    options.dry_tolerance = std::stod(take("dry_tolerance", std::to_string(options.dry_tolerance)));
    options.source_depth = std::stod(take("source_depth", std::to_string(options.source_depth)));
    options.source_velocity_x = std::stod(take("source_u", std::to_string(options.source_velocity_x)));
    options.source_velocity_y = std::stod(take("source_v", std::to_string(options.source_velocity_y)));
    options.output_every = std::stoi(take("output_every", std::to_string(options.output_every)));

    if (!values.empty()) {
        throw std::runtime_error("unknown setting '" + values.begin()->first + "'");
    }
    
    if (options.output_every < 1) {
        throw std::runtime_error("output_every must be at least 1");
    }
    
    if (options.end_time < 0.0 || options.dt_max <= 0.0 || options.gravity <= 0.0
        || options.cfl <= 0.0 || options.cfl > 0.5 || options.dry_tolerance < 0.0
        || options.source_depth < 0.0 || !std::isfinite(options.end_time)
        || !std::isfinite(options.dt_max) || !std::isfinite(options.gravity)
        || !std::isfinite(options.cfl) || !std::isfinite(options.dry_tolerance)
        || !std::isfinite(options.source_depth) || !std::isfinite(options.source_velocity_x)
        || !std::isfinite(options.source_velocity_y)) {
        throw std::runtime_error("invalid numeric options; require 0 < cfl <= 0.5 and valid time/gravity/depth values");
    }
    
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: flow input_file");
        }
        
        const Options options = read_options(argv[1]);
        Mesh mesh = read_mesh(options.nodes_path, options.triangles_path,
                              options.boundaries_path, options.state_path);

        BoundarySeries boundary_series;
        if (!options.boundary_input_path.empty()) {
            boundary_series = read_boundary_series(options.boundary_input_path, options.boundary_input_format == "bed_depth");
            for (const Face& face : mesh.faces) {
                if (face.boundary_tag == 2 && face.right_cell < 0) {
                    (void)sample_boundary(boundary_series, face.midpoint, 0.0);
                }
            }
        }

        std::ofstream boundary_output;
        if (!options.boundary_output_path.empty()) {
            const bool has_outflow = std::any_of(mesh.faces.begin(), mesh.faces.end(),
                [](const Face& face) { return face.boundary_tag == 1; });
            if (!has_outflow) {
                throw std::runtime_error("--boundary-output requires at least one edge tagged 1");
            }
            
            const std::filesystem::path output_path(options.boundary_output_path);
            if (!output_path.parent_path().empty()) {
                std::filesystem::create_directories(output_path.parent_path());
            }
            
            boundary_output.open(output_path);
            if (!boundary_output) {
                throw std::runtime_error("cannot write boundary output: " + options.boundary_output_path);
            }
            
            write_boundary_snapshot(boundary_output, mesh, 0.0, options.dry_tolerance);
        }

        std::vector<Residual> residuals;
        std::vector<BoundaryState> boundary_states(mesh.faces.size(), {
            options.source_depth, options.source_velocity_x, options.source_velocity_y,
        });

#ifdef CUDA
        std::unique_ptr<CudaContext, void (*)(CudaContext*)> cuda_context(nullptr, cuda_destroy);
        std::vector<CellData> cuda_cells;
        std::vector<BoundaryData> cuda_boundary(mesh.faces.size());
        if (options.backend == "cuda") {
            std::vector<FaceData> cuda_faces;
            cuda_faces.reserve(mesh.faces.size());
            for (const Face& face : mesh.faces) {
                cuda_faces.push_back({face.left_cell, face.right_cell, face.boundary_tag, face.normal.x, face.normal.y, face.length});
            }
            
            cuda_cells.reserve(mesh.cells.size());
            for (const Cell& cell : mesh.cells) {
                cuda_cells.push_back({cell.area, cell.bed, cell.bed_dx, cell.bed_dy, cell.depth, cell.depth_x_momentum, cell.depth_y_momentum});
            }
            
            cuda_context.reset(cuda_create(cuda_faces.data(), cuda_faces.size(), mesh.cells.size()));
            if (!cuda_context) {
                throw std::runtime_error("could not initialize CUDA backend");
            }
            
        } 
        
        else if (options.backend != "cpu") {
            throw std::runtime_error("backend must be cpu or cuda");
        }
        
        if (options.backend == "cpu") {
            residuals.resize(mesh.cells.size());
        }
#else
        if (options.backend != "cpu") {
            throw std::runtime_error("no CUDA backend; compile with -DSW_ENABLE_CUDA=ON");
        }
        
        residuals.resize(mesh.cells.size());
#endif

        double time = 0.0;
        double computation_seconds = 0.0;
        int step = 0;
        write(options.output_prefix + "_000000", mesh, time);

        while (time < options.end_time) {
            const auto computation_start = std::chrono::steady_clock::now();
            double time_remaining = options.end_time - time;
            if (!boundary_series.empty()) {
                for (std::size_t i = 0; i < mesh.faces.size(); ++i) {
                    const Face& face = mesh.faces[i];
                    if (face.boundary_tag == 2 && face.right_cell < 0) {
                        boundary_states[i] = sample_boundary(boundary_series, face.midpoint, time);
                        const auto& samples = boundary_series.at(point_key(face.midpoint));
                        const auto next = std::upper_bound(samples.begin(), samples.end(), time, [](double value, const BoundarySample& sample) { return value < sample.time; });
                        if (next != samples.end()) {
                            time_remaining = std::min(time_remaining, next->time - time);
                        }
                    }
                }
            }
            
            const double dt = choose_time_step(
                mesh, time_remaining, options.dt_max,
                options.gravity, options.cfl, options.dry_tolerance, boundary_states);
            if (time + dt == time) {
                throw std::runtime_error("timestep is too small to advance simulation time");
            }

#ifdef CUDA
            if (options.backend == "cuda") {
                for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
                    const Cell& cell = mesh.cells[i];
                    cuda_cells[i].depth = cell.depth;
                    cuda_cells[i].hu = cell.depth_x_momentum;
                    cuda_cells[i].hv = cell.depth_y_momentum;
                }
                
                for (std::size_t i = 0; i < boundary_states.size(); ++i) {
                    const BoundaryState& state = boundary_states[i];
                    cuda_boundary[i] = {state.depth, state.velocity_x, state.velocity_y, state.bed, state.has_bed};
                }
                
                if (!cuda_step(cuda_context.get(), cuda_cells.data(), cuda_boundary.data(), options.gravity, dt, options.dry_tolerance)) {
                    throw std::runtime_error("CUDA step failed");
                }
                
                for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
                    mesh.cells[i].depth = cuda_cells[i].depth;
                    mesh.cells[i].depth_x_momentum = cuda_cells[i].hu;
                    mesh.cells[i].depth_y_momentum = cuda_cells[i].hv;
                }
                
            } else {
#endif
                compute_residuals(mesh, options.gravity, options.dry_tolerance, boundary_states, residuals);
                update_cells(mesh, residuals, dt, options.dry_tolerance);
#ifdef CUDA
            }
#endif
            time += dt;
            ++step;

            const double time_roundoff = 8.0 * std::numeric_limits<double>::epsilon() * std::max(options.end_time, dt) * static_cast<double>(step);
            if (options.end_time - time <= time_roundoff) {
                time = options.end_time;
            }
            
            computation_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - computation_start).count();

            if (step % options.output_every == 0 || time >= options.end_time) {
                const std::string frame_number = std::to_string(step);
                const std::string padded_number(6 - std::min<std::size_t>(6, frame_number.size()), '0');
                const std::string output_path = options.output_prefix + "_" + padded_number
                    + frame_number;
                write(output_path, mesh, time);
                if (!options.boundary_output_path.empty()) {
                    write_boundary_snapshot(boundary_output, mesh, time, options.dry_tolerance);
                }
            }
        }

        double water_volume = 0.0;
        for (const Cell& cell : mesh.cells) {
            water_volume += cell.area * cell.depth;
        }
        
        std::cout << "time=" << time
                  << " cells=" << mesh.cells.size()
                  << " faces=" << mesh.faces.size()
                  << " volume=" << std::setprecision(12) << water_volume
                  << " steps=" << step
                  << " computation_time_seconds=" << computation_seconds << '\n';
                  
        return 0;
    } 
    
    catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        
        return 1;
    }
}
