#include "cuda_backend.h"

#include <cuda_runtime.h>

#include <new>

struct CudaContext {
    FaceData* faces = nullptr;
    BoundaryData* boundary = nullptr;
    CellData* cells = nullptr;
    double* residuals = nullptr;
    std::size_t face_count = 0;
    std::size_t cell_count = 0;
};


namespace {

struct DeviceFlux {
    double depth;
    double x_momentum;
    double y_momentum;
};


__device__ double atomic_add_double(double* address, double value) {
    auto* integer_address = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old_value = *integer_address;
    unsigned long long assumed_value;
    do {
        assumed_value = old_value;
        const double updated = value + __longlong_as_double(assumed_value);
        old_value = atomicCAS(integer_address, assumed_value, __double_as_longlong(updated));
    } while (assumed_value != old_value);
    
    return __longlong_as_double(old_value);
}


__device__ DeviceFlux physical_flux(double depth, double velocity_x, double velocity_y, double nx, double ny, double gravity) {
    const double normal_velocity = velocity_x * nx + velocity_y * ny;
    
    return {
        depth * normal_velocity,
        depth * velocity_x * normal_velocity + 0.5 * gravity * depth * depth * nx,
        depth * velocity_y * normal_velocity + 0.5 * gravity * depth * depth * ny,
    };
}


__global__ void clear_residuals(double* residuals, int cell_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 3 * cell_count) {
        residuals[index] = 0.0;
    }
}


__global__ void compute_face_fluxes(const FaceData* faces, const BoundaryData* boundary, const CellData* cells, double* residuals, int face_count, double gravity, double dry_tolerance) {
    const int face_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_index >= face_count) {
        return;
    }

    const FaceData face = faces[face_index];
    const CellData left_cell = cells[face.left];
    const double left_depth_before_reconstruction = fmax(0.0, left_cell.depth);
    double left_u = left_depth_before_reconstruction > dry_tolerance ? left_cell.hu / left_depth_before_reconstruction : 0.0;
    double left_v = left_depth_before_reconstruction > dry_tolerance ? left_cell.hv / left_depth_before_reconstruction : 0.0;

    double right_depth_before_reconstruction = 0.0;
    double right_u = 0.0;
    double right_v = 0.0;
    if (face.tag == 2 && face.right < 0) {
        right_depth_before_reconstruction = boundary[face_index].depth;
        if (right_depth_before_reconstruction > dry_tolerance) {
            right_u = boundary[face_index].velocity_x;
            right_v = boundary[face_index].velocity_y;
        }
    } 
    
    else if (face.right >= 0 && face.tag != 1) {
        const CellData right_cell = cells[face.right];
        right_depth_before_reconstruction = fmax(0.0, right_cell.depth);
        if (right_depth_before_reconstruction > dry_tolerance) {
            right_u = right_cell.hu / right_depth_before_reconstruction;
            right_v = right_cell.hv / right_depth_before_reconstruction;
        }
    } 
    
    else if (face.tag != 1) {
        const double normal_velocity = left_u * face.nx + left_v * face.ny;
        right_depth_before_reconstruction = left_depth_before_reconstruction;
        right_u = left_u - 2.0 * normal_velocity * face.nx;
        right_v = left_v - 2.0 * normal_velocity * face.ny;
    }

    const BoundaryData source = boundary[face_index];
    const double right_bed = face.tag == 2 && face.right < 0 && source.has_bed ? source.bed : (face.right >= 0 && face.tag != 1 ? cells[face.right].bed : left_cell.bed);
    const double reconstructed_bed = fmax(left_cell.bed, right_bed);
    const double left_depth = fmax(0.0, left_depth_before_reconstruction - (reconstructed_bed - left_cell.bed));
    const double right_depth = fmax(0.0, right_depth_before_reconstruction - (reconstructed_bed - right_bed));
    if (left_depth <= dry_tolerance) {
        left_u = left_v = 0.0;
    }
    
    if (right_depth <= dry_tolerance) {
        right_u = right_v = 0.0;
    }

    const DeviceFlux left_flux = physical_flux(left_depth, left_u, left_v, face.nx, face.ny, gravity);
    const DeviceFlux right_flux = physical_flux(right_depth, right_u, right_v, face.nx, face.ny, gravity);
    const double left_normal_velocity = left_u * face.nx + left_v * face.ny;
    const double right_normal_velocity = right_u * face.nx + right_v * face.ny;
    const double s_left = fmin(left_normal_velocity - sqrt(gravity * left_depth), right_normal_velocity - sqrt(gravity * right_depth));
    const double s_right = fmax(left_normal_velocity + sqrt(gravity * left_depth), right_normal_velocity + sqrt(gravity * right_depth));

    DeviceFlux flux;
    if (s_left >= 0.0) {
        flux = left_flux;
    } 
    
    else if (s_right <= 0.0) {
        flux = right_flux;
    } 
    
    else {
        const double inverse_span = 1.0 / (s_right - s_left);
        flux = {
            (s_right * left_flux.depth - s_left * right_flux.depth + s_left * s_right * (right_depth - left_depth)) * inverse_span,
            (s_right * left_flux.x_momentum - s_left * right_flux.x_momentum  + s_left * s_right * (right_depth * right_u - left_depth * left_u)) * inverse_span,
            (s_right * left_flux.y_momentum - s_left * right_flux.y_momentum  + s_left * s_right * (right_depth * right_v - left_depth * left_v)) * inverse_span,
        };
    }

    const double left_pressure_correction = 0.5 * gravity * (left_depth_before_reconstruction * left_depth_before_reconstruction - left_depth * left_depth);
    atomic_add_double(&residuals[3 * face.left], -flux.depth * face.length);
    atomic_add_double(&residuals[3 * face.left + 1], -(flux.x_momentum + left_pressure_correction * face.nx) * face.length);
    atomic_add_double(&residuals[3 * face.left + 2], -(flux.y_momentum + left_pressure_correction * face.ny) * face.length);

    if (face.right >= 0 && face.tag != 1) {
        const double right_pressure_correction = 0.5 * gravity  * (right_depth_before_reconstruction * right_depth_before_reconstruction - right_depth * right_depth);
        atomic_add_double(&residuals[3 * face.right], flux.depth * face.length);
        atomic_add_double(&residuals[3 * face.right + 1], (flux.x_momentum + right_pressure_correction * face.nx) * face.length);
        atomic_add_double(&residuals[3 * face.right + 2], (flux.y_momentum + right_pressure_correction * face.ny) * face.length);
    }
}

__global__ void update_cells(CellData* cells, const double* residuals, int cell_count, double dt, double dry_tolerance) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cell_count) {
        return;
    }

    CellData cell = cells[index];
    const double new_depth = fmax(0.0, cell.depth + dt * residuals[3 * index] / cell.area);
    cell.hu += dt * residuals[3 * index + 1] / cell.area;
    cell.hv += dt * residuals[3 * index + 2] / cell.area;
    cell.depth = new_depth;
    if (cell.depth <= dry_tolerance) {
        cell.depth = 0.0;
        cell.hu = 0.0;
        cell.hv = 0.0;
    }
    
    cells[index] = cell;
}

bool allocate_device_memory(CudaContext* context) {
    return cudaMalloc(&context->faces, context->face_count * sizeof(FaceData)) == cudaSuccess
        && cudaMalloc(&context->boundary, context->face_count * sizeof(BoundaryData)) == cudaSuccess
        && cudaMalloc(&context->cells, context->cell_count * sizeof(CellData)) == cudaSuccess
        && cudaMalloc(&context->residuals, 3 * context->cell_count * sizeof(double)) == cudaSuccess;
}

} // namespace

CudaContext* cuda_create(const FaceData* faces, std::size_t face_count,
                         std::size_t cell_count) {
    auto* context = new (std::nothrow) CudaContext;
    if (!context) {
        return nullptr;
    }
    
    context->face_count = face_count;
    context->cell_count = cell_count;

    if (!allocate_device_memory(context) || cudaMemcpy(context->faces, faces, face_count * sizeof(FaceData), cudaMemcpyHostToDevice) != cudaSuccess) {
        cuda_destroy(context);
        return nullptr;
    }
    
    return context;
}

bool cuda_step(CudaContext* context, CellData* cells, const BoundaryData* boundary, double gravity, double dt, double dry_tolerance) {
    if (!context || context->face_count == 0 || context->cell_count == 0) {
        return false;
    }
    
    if (cudaMemcpy(context->cells, cells, context->cell_count * sizeof(CellData), cudaMemcpyHostToDevice) != cudaSuccess
        || cudaMemcpy(context->boundary, boundary, context->face_count * sizeof(BoundaryData), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    constexpr int threads_per_block = 256;
    const int cell_blocks = static_cast<int>(
        (3 * context->cell_count + threads_per_block - 1) / threads_per_block);
    const int face_blocks = static_cast<int>(
        (context->face_count + threads_per_block - 1) / threads_per_block);
    const int update_blocks = static_cast<int>(
        (context->cell_count + threads_per_block - 1) / threads_per_block);

    clear_residuals<<<cell_blocks, threads_per_block>>>(context->residuals, static_cast<int>(context->cell_count));
    compute_face_fluxes<<<face_blocks, threads_per_block>>>(context->faces, context->boundary, context->cells, context->residuals, static_cast<int>(context->face_count), gravity, dry_tolerance);
    update_cells<<<update_blocks, threads_per_block>>>(context->cells, context->residuals, static_cast<int>(context->cell_count), dt, dry_tolerance);

    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        return false;
    }
    
    return cudaMemcpy(cells, context->cells, context->cell_count * sizeof(CellData), cudaMemcpyDeviceToHost) == cudaSuccess;
}

void cuda_destroy(CudaContext* context) {
    if (!context) {
        return;
    }
    
    if (context->faces) cudaFree(context->faces);
    if (context->boundary) cudaFree(context->boundary);
    if (context->cells) cudaFree(context->cells);
    if (context->residuals) cudaFree(context->residuals);
    
    delete context;
}
