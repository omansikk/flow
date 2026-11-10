#pragma once
#include <cstddef>


struct FaceData {
    int left;
    int right;
    int tag;
    double nx;
    double ny;
    double length;
};


struct CellData {
    double area;
    double bed;
    double bed_dx;
    double bed_dy;
    double depth;
    double hu;
    double hv;
};


struct BoundaryData {
    double depth;
    double velocity_x;
    double velocity_y;
    double bed;
    bool has_bed;
};


struct CudaContext;

CudaContext* cuda_create(const FaceData* faces, std::size_t face_count, std::size_t cell_count);

bool cuda_step(CudaContext* context, CellData* cells, const BoundaryData* boundary, double gravity, double dt, double dry_tolerance);

void cuda_destroy(CudaContext* context);
