#include <cuda.h>
#include <cuda_runtime_api.h>
#include "ss_data.h"
#include "dspaces-common.h"

#define CUDA_ASSERT(x)                                                          \
    do                                                                          \
        {                                                                       \
            if (!(x))                                                           \
                {                                                               \
                    fprintf(stderr, "%s, line %i (%s):"                \
                            "Assertion %s failed!\n",                           \
                            __FILE__, __LINE__, __func__, #x);    \
                    return dspaces_ERR_CUDA;                                    \
                }                                                               \
        } while (0)

#define CUDA_ASSERTRT(stmt)				                                        \
    do                                                                          \
        {                                                                       \
            cudaError_t err = (stmt);                                           \
            if (err != cudaSuccess) {                                           \
                fprintf(stderr, "%s, line %i (%s):"                    \
                        "%s failed, Err Code: (%s)\n",                          \
                        __FILE__, __LINE__, __func__, #stmt,      \
                        cudaGetErrorString(err));                               \
            }                                                                   \
            CUDA_ASSERT(cudaSuccess == err);                                \
        } while (0)

__global__ void copy_subarray_c_double(double *dst, double *src, int dst_nx, int dst_ny, int dst_nz,
    int src_nx, int src_ny, int src_nz, int sub_nx, int sub_ny, int sub_nz)
{
    //==============================================================================
    // 2 Registers | 3 arguments
    //==============================================================================
    int   i, j, k;
    //==============================================================================

    // Identify current thread
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
        dst[i * dst_ny * dst_nz + j * dst_nz + k] = src[i * src_ny * src_nz + j * src_nz + k];
    }
}

__global__ void copy_subarray_c_float(float *dst, float *src, int dst_nx, int dst_ny, int dst_nz,
    int src_nx, int src_ny, int src_nz, int sub_nx, int sub_ny, int sub_nz)
{
    //==============================================================================
    // 2 Registers | 3 arguments
    //==============================================================================
    int   i, j, k;
    //==============================================================================

    // Identify current thread
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
        dst[i * dst_ny * dst_nz + j * dst_nz + k] = src[i * src_ny * src_nz + j * src_nz + k];
    }
}

__global__ void copy_subarray_c_short(short *dst, short *src, int dst_nx, int dst_ny, int dst_nz,
    int src_nx, int src_ny, int src_nz, int sub_nx, int sub_ny, int sub_nz)
{
    //==============================================================================
    // 2 Registers | 3 arguments
    //==============================================================================
    int   i, j, k;
    //==============================================================================

    // Identify current thread
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
        dst[i * dst_ny * dst_nz + j * dst_nz + k] = src[i * src_ny * src_nz + j * src_nz + k];
    }
}

__global__ void copy_subarray_c_double(char *dst, char *src, int dst_nx, int dst_ny, int dst_nz,
    int src_nx, int src_ny, int src_nz, int sub_nx, int sub_ny, int sub_nz)
{
    //==============================================================================
    // 2 Registers | 3 arguments
    //==============================================================================
    int   i, j, k;
    //==============================================================================

    // Identify current thread
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
        dst[i * dst_ny * dst_nz + j * dst_nz + k] = src[i * src_ny * src_nz + j * src_nz + k];
    }
}

__global__ void copy_subarray_f_double(double *dst, double *src, int dst_nx, int dst_ny, int dst_nz,
                                    int src_nx, int src_ny, int src_nz, int sub_nx, int sub_ny, int sub_nz)
{
    //==============================================================================
    // 2 Registers | 3 arguments
    //==============================================================================
    int   i, j, k;
    //==============================================================================

    // Identify current thread
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
        dst[i + j * dst_nx + k * dst_nx * dst_ny] = src[i + j * src_nx + k * src_nx * src_ny];
    }
}

__global__ void copy_subarray_f_float(float *dst, float *src, int dst_nx, int dst_ny, int dst_nz,
    int src_nx, int src_ny, int src_nz, int sub_nx, int sub_ny, int sub_nz)
{
    //==============================================================================
    // 2 Registers | 3 arguments
    //==============================================================================
    int   i, j, k;
    //==============================================================================

    // Identify current thread
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
        dst[i + j * dst_nx + k * dst_nx * dst_ny] = src[i + j * src_nx + k * src_nx * src_ny];
    }
}

__global__ void copy_subarray_f_short(short *dst, short *src, int dst_nx, int dst_ny, int dst_nz,
    int src_nx, int src_ny, int src_nz, int sub_nx, int sub_ny, int sub_nz)
{
    //==============================================================================
    // 2 Registers | 3 arguments
    //==============================================================================
    int   i, j, k;
    //==============================================================================

    // Identify current thread
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
        dst[i + j * dst_nx + k * dst_nx * dst_ny] = src[i + j * src_nx + k * src_nx * src_ny];
    }
}

__global__ void copy_subarray_f_char(char *dst, char *src, int dst_nx, int dst_ny, int dst_nz,
    int src_nx, int src_ny, int src_nz, int sub_nx, int sub_ny, int sub_nz)
{
    //==============================================================================
    // 2 Registers | 3 arguments
    //==============================================================================
    int   i, j, k;
    //==============================================================================

    // Identify current thread
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    if ( (i < sub_nx) && (j < sub_ny) && (k < sub_nz) ) {
        dst[i + j * dst_nx + k * dst_nx * dst_ny] = src[i + j * src_nx + k * src_nx * src_ny];
    }
}

extern "C" int matrix_copy_cuda_f_double(struct matrix *dst, struct matrix *src)
{
    double *d = (double*) dst->pdata;
    double *s = (double*) src->pdata;

    // int BLOCK_THREAD_SIZE = 1024;
    int BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z;
    int sub_nx, sub_ny, sub_nz;
    int GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z;

    uint64_t dst9, dst8, dst7, dst6, dst5, dst4, dst3;
    uint64_t dst_off9 = 0, dst_off8 = 0, dst_off7 = 0,
             dst_off6 = 0, dst_off5 = 0, dst_off4 = 0;
    uint64_t src9, src8, src7, src6, src5, src4, src3;
    uint64_t src_off9 = 0, src_off8 = 0, src_off7 = 0,
             src_off6 = 0, src_off5 = 0, src_off4 = 0;
    uint64_t dst_off3, src_off3;
    uint64_t dst_off, src_off; 
    uint64_t dst_stride3, src_stride3;

    // Use non-parallel design for unit benchmark
    cudaStream_t stream;
    CUDA_ASSERTRT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if(dst->num_dims == 1) {
        BLOCK_SIZE_X = 256; 
        BLOCK_SIZE_Y = 4;
        BLOCK_SIZE_Z = 1;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = 1;
        sub_nz = 1;
        dst_off = dst->mat_view.lb[0];
        src_off = src->mat_view.lb[0];
    } else if(dst->num_dims == 2) {
        BLOCK_SIZE_X = 16;
        BLOCK_SIZE_Y = 32;
        BLOCK_SIZE_Z = 1;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = dst->mat_view.ub[1] - dst->mat_view.lb[1] + 1;
        sub_nz = 1;
        dst_off = dst->mat_view.lb[1] * dst->dist[0] + dst->mat_view.lb[0];
        src_off = src->mat_view.lb[1] * src->dist[0] + src->mat_view.lb[0];
    } else { 
        // ndims >= 3 will use 3D kernel in loops, so the params are the same
        BLOCK_SIZE_X = 8;
        BLOCK_SIZE_Y = 8;
        BLOCK_SIZE_Z = 16;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = dst->mat_view.ub[1] - dst->mat_view.lb[1] + 1;
        sub_nz = dst->mat_view.ub[2] - dst->mat_view.lb[2] + 1;

        dst_off3 = (dst->mat_view.lb[2] * dst->dist[1] + dst->mat_view.lb[1]) * dst->dist[0] + dst->mat_view.lb[0];
        src_off3 = (src->mat_view.lb[2] * src->dist[1] + src->mat_view.lb[1]) * src->dist[0] + src->mat_view.lb[0];

        dst_stride3 = dst->dist[2] * dst->dist[1] * dst->dist[0];
        src_stride3 = src->dist[2] * dst->dist[1] * dst->dist[0];

        // only ndims == 3 use fixed dst & src offset, others will change the values as excuted in loops
        if(dst->num_dims ==3) {
            dst_off = dst_off3;
            src_off = src_off3;
        }
    }

    GRID_SIZE_X = (sub_nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    GRID_SIZE_Y = (sub_ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    GRID_SIZE_Z = (sub_nz + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z;
    
    dim3 dimblock( BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z );
    dim3 dimgrid( GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z );

    switch(dst->num_dims) {
        case 1:
            goto ndimleq3;
            break;
        case 2:
            goto ndimleq3;
            break;
        case 3:
            goto ndimleq3;
            break;
        case 4:
            goto ndim4;
            break;
        case 5:
            goto ndim5;
            break;
        case 6:
            goto ndim6;
            break;
        case 7:
            goto ndim7;
            break;
        case 8:
            goto ndim8;
            break;
        case 9:
            goto ndim9;
            break;
        case 10:
            goto ndim10;
            break;
        default:
            return dspaces_ERR_INVALID_ARG;
            break;
        }

ndim10:
    for(dst9 = dst->mat_view.lb[9], src9 = src->mat_view.lb[9];
        dst9 <= dst->mat_view.ub[9]; dst9++, src9++) {
        dst_off9 = dst9 * dst->dist[8];
        src_off9 = src9 * src->dist[8];
    ndim9:
        for(dst8 = dst->mat_view.lb[8], src8 = src->mat_view.lb[8];
            dst8 <= dst->mat_view.ub[8]; dst8++, src8++) {
            dst_off8 = (dst_off9 + dst8) * dst->dist[7];
            src_off8 = (src_off9 + src8) * src->dist[7];
        ndim8:
            for(dst7 = dst->mat_view.lb[7], src7 = src->mat_view.lb[7];
                dst7 <= dst->mat_view.ub[7]; dst7++, src7++) {
                dst_off7 = (dst_off8 + dst7) * dst->dist[6];
                src_off7 = (src_off8 + src7) * src->dist[6];
            ndim7:
                for(dst6 = dst->mat_view.lb[6], src6 = src->mat_view.lb[6];
                    dst6 <= dst->mat_view.ub[6]; dst6++, src6++) {
                    dst_off6 = (dst_off7 + dst6) * dst->dist[5];
                    src_off6 = (src_off7 + src6) * src->dist[5];
                ndim6:
                    for(dst5 = dst->mat_view.lb[5], src5 = src->mat_view.lb[5];
                        dst5 <= dst->mat_view.ub[5]; dst5++, src5++) {
                        dst_off5 = (dst_off6 + dst5) * dst->dist[4];
                        src_off5 = (src_off6 + src5) * src->dist[4];
                    ndim5:
                        for(dst4 = dst->mat_view.lb[4], src4 = src->mat_view.lb[4];
                            dst4 <= dst->mat_view.ub[4]; dst4++, src4++) {
                            dst_off4 = (dst_off5 + dst4) * dst->dist[3];
                            src_off4 = (src_off5 + src4) * src->dist[3];
                        ndim4:
                            for(dst3 = dst->mat_view.lb[3], src3 = src->mat_view.lb[3];
                                dst3 <= dst->mat_view.ub[3]; dst3++, src3++) {
                                dst_off = (dst_off4 + dst3) * dst_stride3 + dst_off3;
                                src_off = (src_off4 + src3) * src_stride3 + dst_off3;
                            ndimleq3:
                                copy_subarray_f_double<<<dimgrid, dimblock, 0, stream>>>(&d[dst_off],
                                    &s[src_off], dst->dist[0], dst->dist[1], dst->dist[2],
                                    src->dist[0], src->dist[1], src->dist[2], sub_nx, sub_ny, sub_nz);
                                CUDA_ASSERTRT(cudaStreamSynchronize(stream));
                                if(src->num_dims <= 3)
                                    return dspaces_SUCCESS;
                            }
                            if(src->num_dims == 4)
                                return dspaces_SUCCESS;
                        }
                        if(src->num_dims == 5)
                            return dspaces_SUCCESS;
                    }
                    if(src->num_dims == 6)
                        return dspaces_SUCCESS;
                }
                if(src->num_dims == 7)
                    return dspaces_SUCCESS;
            }
            if(src->num_dims == 8)
                return dspaces_SUCCESS;
        }
        if(src->num_dims == 9)
            return dspaces_SUCCESS;
    }
    return dspaces_SUCCESS;
}

extern "C" int matrix_copy_cuda_f_float(struct matrix *dst, struct matrix *src)
{
    float *d = (float*) dst->pdata;
    float *s = (float*) src->pdata;

    // int BLOCK_THREAD_SIZE = 1024;
    int BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z;
    int sub_nx, sub_ny, sub_nz;
    int GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z;

    uint64_t dst9, dst8, dst7, dst6, dst5, dst4, dst3;
    uint64_t dst_off9 = 0, dst_off8 = 0, dst_off7 = 0,
             dst_off6 = 0, dst_off5 = 0, dst_off4 = 0;
    uint64_t src9, src8, src7, src6, src5, src4, src3;
    uint64_t src_off9 = 0, src_off8 = 0, src_off7 = 0,
             src_off6 = 0, src_off5 = 0, src_off4 = 0;
    uint64_t dst_off3, src_off3;
    uint64_t dst_off, src_off; 
    uint64_t dst_stride3, src_stride3;

    // Use non-parallel design for unit benchmark
    cudaStream_t stream;
    CUDA_ASSERTRT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if(dst->num_dims == 1) {
        BLOCK_SIZE_X = 256; 
        BLOCK_SIZE_Y = 4;
        BLOCK_SIZE_Z = 1;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = 1;
        sub_nz = 1;
        dst_off = dst->mat_view.lb[0];
        src_off = src->mat_view.lb[0];
    } else if(dst->num_dims == 2) {
        BLOCK_SIZE_X = 16;
        BLOCK_SIZE_Y = 32;
        BLOCK_SIZE_Z = 1;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = dst->mat_view.ub[1] - dst->mat_view.lb[1] + 1;
        sub_nz = 1;
        dst_off = dst->mat_view.lb[1] * dst->dist[0] + dst->mat_view.lb[0];
        src_off = src->mat_view.lb[1] * src->dist[0] + src->mat_view.lb[0];
    } else { 
        // ndims >= 3 will use 3D kernel in loops, so the params are the same
        BLOCK_SIZE_X = 8;
        BLOCK_SIZE_Y = 8;
        BLOCK_SIZE_Z = 16;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = dst->mat_view.ub[1] - dst->mat_view.lb[1] + 1;
        sub_nz = dst->mat_view.ub[2] - dst->mat_view.lb[2] + 1;

        dst_off3 = (dst->mat_view.lb[2] * dst->dist[1] + dst->mat_view.lb[1]) * dst->dist[0] + dst->mat_view.lb[0];
        src_off3 = (src->mat_view.lb[2] * src->dist[1] + src->mat_view.lb[1]) * src->dist[0] + src->mat_view.lb[0];

        dst_stride3 = dst->dist[2] * dst->dist[1] * dst->dist[0];
        src_stride3 = src->dist[2] * dst->dist[1] * dst->dist[0];

        // only ndims == 3 use fixed dst & src offset, others will change the values as excuted in loops
        if(dst->num_dims ==3) {
            dst_off = dst_off3;
            src_off = src_off3;
        }
    }

    GRID_SIZE_X = (sub_nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    GRID_SIZE_Y = (sub_ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    GRID_SIZE_Z = (sub_nz + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z;
    
    dim3 dimblock( BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z );
    dim3 dimgrid( GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z );

    switch(dst->num_dims) {
        case 1:
            goto ndimleq3;
            break;
        case 2:
            goto ndimleq3;
            break;
        case 3:
            goto ndimleq3;
            break;
        case 4:
            goto ndim4;
            break;
        case 5:
            goto ndim5;
            break;
        case 6:
            goto ndim6;
            break;
        case 7:
            goto ndim7;
            break;
        case 8:
            goto ndim8;
            break;
        case 9:
            goto ndim9;
            break;
        case 10:
            goto ndim10;
            break;
        default:
            return dspaces_ERR_INVALID_ARG;
            break;
        }

ndim10:
    for(dst9 = dst->mat_view.lb[9], src9 = src->mat_view.lb[9];
        dst9 <= dst->mat_view.ub[9]; dst9++, src9++) {
        dst_off9 = dst9 * dst->dist[8];
        src_off9 = src9 * src->dist[8];
    ndim9:
        for(dst8 = dst->mat_view.lb[8], src8 = src->mat_view.lb[8];
            dst8 <= dst->mat_view.ub[8]; dst8++, src8++) {
            dst_off8 = (dst_off9 + dst8) * dst->dist[7];
            src_off8 = (src_off9 + src8) * src->dist[7];
        ndim8:
            for(dst7 = dst->mat_view.lb[7], src7 = src->mat_view.lb[7];
                dst7 <= dst->mat_view.ub[7]; dst7++, src7++) {
                dst_off7 = (dst_off8 + dst7) * dst->dist[6];
                src_off7 = (src_off8 + src7) * src->dist[6];
            ndim7:
                for(dst6 = dst->mat_view.lb[6], src6 = src->mat_view.lb[6];
                    dst6 <= dst->mat_view.ub[6]; dst6++, src6++) {
                    dst_off6 = (dst_off7 + dst6) * dst->dist[5];
                    src_off6 = (src_off7 + src6) * src->dist[5];
                ndim6:
                    for(dst5 = dst->mat_view.lb[5], src5 = src->mat_view.lb[5];
                        dst5 <= dst->mat_view.ub[5]; dst5++, src5++) {
                        dst_off5 = (dst_off6 + dst5) * dst->dist[4];
                        src_off5 = (src_off6 + src5) * src->dist[4];
                    ndim5:
                        for(dst4 = dst->mat_view.lb[4], src4 = src->mat_view.lb[4];
                            dst4 <= dst->mat_view.ub[4]; dst4++, src4++) {
                            dst_off4 = (dst_off5 + dst4) * dst->dist[3];
                            src_off4 = (src_off5 + src4) * src->dist[3];
                        ndim4:
                            for(dst3 = dst->mat_view.lb[3], src3 = src->mat_view.lb[3];
                                dst3 <= dst->mat_view.ub[3]; dst3++, src3++) {
                                dst_off = (dst_off4 + dst3) * dst_stride3 + dst_off3;
                                src_off = (src_off4 + src3) * src_stride3 + dst_off3;
                            ndimleq3:
                                copy_subarray_f_float<<<dimgrid, dimblock, 0, stream>>>(&d[dst_off],
                                    &s[src_off], dst->dist[0], dst->dist[1], dst->dist[2],
                                    src->dist[0], src->dist[1], src->dist[2], sub_nx, sub_ny, sub_nz);
                                CUDA_ASSERTRT(cudaStreamSynchronize(stream));
                                if(src->num_dims <= 3)
                                    return dspaces_SUCCESS;
                            }
                            if(src->num_dims == 4)
                                return dspaces_SUCCESS;
                        }
                        if(src->num_dims == 5)
                            return dspaces_SUCCESS;
                    }
                    if(src->num_dims == 6)
                        return dspaces_SUCCESS;
                }
                if(src->num_dims == 7)
                    return dspaces_SUCCESS;
            }
            if(src->num_dims == 8)
                return dspaces_SUCCESS;
        }
        if(src->num_dims == 9)
            return dspaces_SUCCESS;
    }
    return dspaces_SUCCESS;
}

extern "C" int matrix_copy_cuda_f_short(struct matrix *dst, struct matrix *src)
{
    short *d = (short*) dst->pdata;
    short *s = (short*) src->pdata;

    // int BLOCK_THREAD_SIZE = 1024;
    int BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z;
    int sub_nx, sub_ny, sub_nz;
    int GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z;

    uint64_t dst9, dst8, dst7, dst6, dst5, dst4, dst3;
    uint64_t dst_off9 = 0, dst_off8 = 0, dst_off7 = 0,
             dst_off6 = 0, dst_off5 = 0, dst_off4 = 0;
    uint64_t src9, src8, src7, src6, src5, src4, src3;
    uint64_t src_off9 = 0, src_off8 = 0, src_off7 = 0,
             src_off6 = 0, src_off5 = 0, src_off4 = 0;
    uint64_t dst_off3, src_off3;
    uint64_t dst_off, src_off; 
    uint64_t dst_stride3, src_stride3;

    // Use non-parallel design for unit benchmark
    cudaStream_t stream;
    CUDA_ASSERTRT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if(dst->num_dims == 1) {
        BLOCK_SIZE_X = 256; 
        BLOCK_SIZE_Y = 4;
        BLOCK_SIZE_Z = 1;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = 1;
        sub_nz = 1;
        dst_off = dst->mat_view.lb[0];
        src_off = src->mat_view.lb[0];
    } else if(dst->num_dims == 2) {
        BLOCK_SIZE_X = 16;
        BLOCK_SIZE_Y = 32;
        BLOCK_SIZE_Z = 1;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = dst->mat_view.ub[1] - dst->mat_view.lb[1] + 1;
        sub_nz = 1;
        dst_off = dst->mat_view.lb[1] * dst->dist[0] + dst->mat_view.lb[0];
        src_off = src->mat_view.lb[1] * src->dist[0] + src->mat_view.lb[0];
    } else { 
        // ndims >= 3 will use 3D kernel in loops, so the params are the same
        BLOCK_SIZE_X = 8;
        BLOCK_SIZE_Y = 8;
        BLOCK_SIZE_Z = 16;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = dst->mat_view.ub[1] - dst->mat_view.lb[1] + 1;
        sub_nz = dst->mat_view.ub[2] - dst->mat_view.lb[2] + 1;

        dst_off3 = (dst->mat_view.lb[2] * dst->dist[1] + dst->mat_view.lb[1]) * dst->dist[0] + dst->mat_view.lb[0];
        src_off3 = (src->mat_view.lb[2] * src->dist[1] + src->mat_view.lb[1]) * src->dist[0] + src->mat_view.lb[0];

        dst_stride3 = dst->dist[2] * dst->dist[1] * dst->dist[0];
        src_stride3 = src->dist[2] * dst->dist[1] * dst->dist[0];

        // only ndims == 3 use fixed dst & src offset, others will change the values as excuted in loops
        if(dst->num_dims ==3) {
            dst_off = dst_off3;
            src_off = src_off3;
        }
    }

    GRID_SIZE_X = (sub_nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    GRID_SIZE_Y = (sub_ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    GRID_SIZE_Z = (sub_nz + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z;
    
    dim3 dimblock( BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z );
    dim3 dimgrid( GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z );

    switch(dst->num_dims) {
        case 1:
            goto ndimleq3;
            break;
        case 2:
            goto ndimleq3;
            break;
        case 3:
            goto ndimleq3;
            break;
        case 4:
            goto ndim4;
            break;
        case 5:
            goto ndim5;
            break;
        case 6:
            goto ndim6;
            break;
        case 7:
            goto ndim7;
            break;
        case 8:
            goto ndim8;
            break;
        case 9:
            goto ndim9;
            break;
        case 10:
            goto ndim10;
            break;
        default:
            return dspaces_ERR_INVALID_ARG;
            break;
        }

ndim10:
    for(dst9 = dst->mat_view.lb[9], src9 = src->mat_view.lb[9];
        dst9 <= dst->mat_view.ub[9]; dst9++, src9++) {
        dst_off9 = dst9 * dst->dist[8];
        src_off9 = src9 * src->dist[8];
    ndim9:
        for(dst8 = dst->mat_view.lb[8], src8 = src->mat_view.lb[8];
            dst8 <= dst->mat_view.ub[8]; dst8++, src8++) {
            dst_off8 = (dst_off9 + dst8) * dst->dist[7];
            src_off8 = (src_off9 + src8) * src->dist[7];
        ndim8:
            for(dst7 = dst->mat_view.lb[7], src7 = src->mat_view.lb[7];
                dst7 <= dst->mat_view.ub[7]; dst7++, src7++) {
                dst_off7 = (dst_off8 + dst7) * dst->dist[6];
                src_off7 = (src_off8 + src7) * src->dist[6];
            ndim7:
                for(dst6 = dst->mat_view.lb[6], src6 = src->mat_view.lb[6];
                    dst6 <= dst->mat_view.ub[6]; dst6++, src6++) {
                    dst_off6 = (dst_off7 + dst6) * dst->dist[5];
                    src_off6 = (src_off7 + src6) * src->dist[5];
                ndim6:
                    for(dst5 = dst->mat_view.lb[5], src5 = src->mat_view.lb[5];
                        dst5 <= dst->mat_view.ub[5]; dst5++, src5++) {
                        dst_off5 = (dst_off6 + dst5) * dst->dist[4];
                        src_off5 = (src_off6 + src5) * src->dist[4];
                    ndim5:
                        for(dst4 = dst->mat_view.lb[4], src4 = src->mat_view.lb[4];
                            dst4 <= dst->mat_view.ub[4]; dst4++, src4++) {
                            dst_off4 = (dst_off5 + dst4) * dst->dist[3];
                            src_off4 = (src_off5 + src4) * src->dist[3];
                        ndim4:
                            for(dst3 = dst->mat_view.lb[3], src3 = src->mat_view.lb[3];
                                dst3 <= dst->mat_view.ub[3]; dst3++, src3++) {
                                dst_off = (dst_off4 + dst3) * dst_stride3 + dst_off3;
                                src_off = (src_off4 + src3) * src_stride3 + dst_off3;
                            ndimleq3:
                                copy_subarray_f_short<<<dimgrid, dimblock, 0, stream>>>(&d[dst_off],
                                    &s[src_off], dst->dist[0], dst->dist[1], dst->dist[2],
                                    src->dist[0], src->dist[1], src->dist[2], sub_nx, sub_ny, sub_nz);
                                CUDA_ASSERTRT(cudaStreamSynchronize(stream));
                                if(src->num_dims <= 3)
                                    return dspaces_SUCCESS;
                            }
                            if(src->num_dims == 4)
                                return dspaces_SUCCESS;
                        }
                        if(src->num_dims == 5)
                            return dspaces_SUCCESS;
                    }
                    if(src->num_dims == 6)
                        return dspaces_SUCCESS;
                }
                if(src->num_dims == 7)
                    return dspaces_SUCCESS;
            }
            if(src->num_dims == 8)
                return dspaces_SUCCESS;
        }
        if(src->num_dims == 9)
            return dspaces_SUCCESS;
    }
    return dspaces_SUCCESS;
}

extern "C" int matrix_copy_cuda_f_char(struct matrix *dst, struct matrix *src)
{
    char *d = (char*) dst->pdata;
    char *s = (char*) src->pdata;
    
    // int BLOCK_THREAD_SIZE = 1024;
    int BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z;
    int sub_nx, sub_ny, sub_nz;
    int GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z;

    uint64_t dst9, dst8, dst7, dst6, dst5, dst4, dst3;
    uint64_t dst_off9 = 0, dst_off8 = 0, dst_off7 = 0,
             dst_off6 = 0, dst_off5 = 0, dst_off4 = 0;
    uint64_t src9, src8, src7, src6, src5, src4, src3;
    uint64_t src_off9 = 0, src_off8 = 0, src_off7 = 0,
             src_off6 = 0, src_off5 = 0, src_off4 = 0;
    uint64_t dst_off3, src_off3;
    uint64_t dst_off, src_off; 
    uint64_t dst_stride3, src_stride3;

    // Use non-parallel design for unit benchmark
    cudaStream_t stream;
    CUDA_ASSERTRT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // char function is used for arbitrary data types
    // Therefore, it needs to multiply elem_size to calculate the offsets and copy sizes
    if(dst->num_dims == 1) {
        BLOCK_SIZE_X = 256; 
        BLOCK_SIZE_Y = 4;
        BLOCK_SIZE_Z = 1;
        sub_nx = (dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1) * dst->size_elem;
        sub_ny = 1 * dst->size_elem;
        sub_nz = 1 * dst->size_elem;
        dst_off = dst->mat_view.lb[0] * dst->size_elem;
        src_off = src->mat_view.lb[0] * dst->size_elem;
    } else if(dst->num_dims == 2) {
        BLOCK_SIZE_X = 16;
        BLOCK_SIZE_Y = 32;
        BLOCK_SIZE_Z = 1;
        sub_nx = (dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1) * dst->size_elem;
        sub_ny = (dst->mat_view.ub[1] - dst->mat_view.lb[1] + 1) * dst->size_elem;
        sub_nz = 1 * dst->size_elem;
        dst_off = (dst->mat_view.lb[1] * dst->dist[0] + dst->mat_view.lb[0]) * dst->size_elem;
        src_off = (src->mat_view.lb[1] * src->dist[0] + src->mat_view.lb[0]) * dst->size_elem;
    } else { 
        // ndims >= 3 will use 3D kernel in loops, so the params are the same
        BLOCK_SIZE_X = 8;
        BLOCK_SIZE_Y = 8;
        BLOCK_SIZE_Z = 16;
        sub_nx = (dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1) * dst->size_elem;
        sub_ny = (dst->mat_view.ub[1] - dst->mat_view.lb[1] + 1) * dst->size_elem;
        sub_nz = (dst->mat_view.ub[2] - dst->mat_view.lb[2] + 1) * dst->size_elem;

        dst_off3 = ((dst->mat_view.lb[2] * dst->dist[1] + dst->mat_view.lb[1]) * dst->dist[0] + dst->mat_view.lb[0]) * dst->size_elem;
        src_off3 = ((src->mat_view.lb[2] * src->dist[1] + src->mat_view.lb[1]) * src->dist[0] + src->mat_view.lb[0]) * dst->size_elem;

        dst_stride3 = (dst->dist[2] * dst->dist[1] * dst->dist[0]) * dst->size_elem;
        src_stride3 = (src->dist[2] * dst->dist[1] * dst->dist[0]) * dst->size_elem;

        // only ndims == 3 use fixed dst & src offset, others will change the values as excuted in loops
        if(dst->num_dims ==3) {
            dst_off = dst_off3;
            src_off = src_off3;
        }
    }

    GRID_SIZE_X = (sub_nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    GRID_SIZE_Y = (sub_ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    GRID_SIZE_Z = (sub_nz + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z;
    
    dim3 dimblock( BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z );
    dim3 dimgrid( GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z );

    switch(dst->num_dims) {
        case 1:
            goto ndimleq3;
            break;
        case 2:
            goto ndimleq3;
            break;
        case 3:
            goto ndimleq3;
            break;
        case 4:
            goto ndim4;
            break;
        case 5:
            goto ndim5;
            break;
        case 6:
            goto ndim6;
            break;
        case 7:
            goto ndim7;
            break;
        case 8:
            goto ndim8;
            break;
        case 9:
            goto ndim9;
            break;
        case 10:
            goto ndim10;
            break;
        default:
            return dspaces_ERR_INVALID_ARG;
            break;
        }

ndim10:
    for(dst9 = dst->mat_view.lb[9], src9 = src->mat_view.lb[9];
        dst9 <= dst->mat_view.ub[9]; dst9++, src9++) {
        dst_off9 = dst9 * dst->dist[8];
        src_off9 = src9 * src->dist[8];
    ndim9:
        for(dst8 = dst->mat_view.lb[8], src8 = src->mat_view.lb[8];
            dst8 <= dst->mat_view.ub[8]; dst8++, src8++) {
            dst_off8 = (dst_off9 + dst8) * dst->dist[7];
            src_off8 = (src_off9 + src8) * src->dist[7];
        ndim8:
            for(dst7 = dst->mat_view.lb[7], src7 = src->mat_view.lb[7];
                dst7 <= dst->mat_view.ub[7]; dst7++, src7++) {
                dst_off7 = (dst_off8 + dst7) * dst->dist[6];
                src_off7 = (src_off8 + src7) * src->dist[6];
            ndim7:
                for(dst6 = dst->mat_view.lb[6], src6 = src->mat_view.lb[6];
                    dst6 <= dst->mat_view.ub[6]; dst6++, src6++) {
                    dst_off6 = (dst_off7 + dst6) * dst->dist[5];
                    src_off6 = (src_off7 + src6) * src->dist[5];
                ndim6:
                    for(dst5 = dst->mat_view.lb[5], src5 = src->mat_view.lb[5];
                        dst5 <= dst->mat_view.ub[5]; dst5++, src5++) {
                        dst_off5 = (dst_off6 + dst5) * dst->dist[4];
                        src_off5 = (src_off6 + src5) * src->dist[4];
                    ndim5:
                        for(dst4 = dst->mat_view.lb[4], src4 = src->mat_view.lb[4];
                            dst4 <= dst->mat_view.ub[4]; dst4++, src4++) {
                            dst_off4 = (dst_off5 + dst4) * dst->dist[3];
                            src_off4 = (src_off5 + src4) * src->dist[3];
                        ndim4:
                            for(dst3 = dst->mat_view.lb[3], src3 = src->mat_view.lb[3];
                                dst3 <= dst->mat_view.ub[3]; dst3++, src3++) {
                                dst_off = (dst_off4 + dst3) * dst_stride3 + dst_off3;
                                src_off = (src_off4 + src3) * src_stride3 + dst_off3;
                            ndimleq3:
                                copy_subarray_f_char<<<dimgrid, dimblock, 0, stream>>>(&d[dst_off],
                                    &s[src_off], dst->dist[0], dst->dist[1], dst->dist[2],
                                    src->dist[0], src->dist[1], src->dist[2], sub_nx, sub_ny, sub_nz);
                                CUDA_ASSERTRT(cudaStreamSynchronize(stream));
                                if(src->num_dims <= 3)
                                    return dspaces_SUCCESS;
                            }
                            if(src->num_dims == 4)
                                return dspaces_SUCCESS;
                        }
                        if(src->num_dims == 5)
                            return dspaces_SUCCESS;
                    }
                    if(src->num_dims == 6)
                        return dspaces_SUCCESS;
                }
                if(src->num_dims == 7)
                    return dspaces_SUCCESS;
            }
            if(src->num_dims == 8)
                return dspaces_SUCCESS;
        }
        if(src->num_dims == 9)
            return dspaces_SUCCESS;
    }
    return dspaces_SUCCESS;
}

extern "C" int matrix_copy_cuda_f_double_async(struct matrix *dst, struct matrix *src, cudaStream_t *stream)
{
    double *d = (double*) dst->pdata;
    double *s = (double*) src->pdata;

    // int BLOCK_THREAD_SIZE = 1024;
    int BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z;
    int sub_nx, sub_ny, sub_nz;
    int GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z;

    uint64_t dst9, dst8, dst7, dst6, dst5, dst4, dst3;
    uint64_t dst_off9 = 0, dst_off8 = 0, dst_off7 = 0,
             dst_off6 = 0, dst_off5 = 0, dst_off4 = 0;
    uint64_t src9, src8, src7, src6, src5, src4, src3;
    uint64_t src_off9 = 0, src_off8 = 0, src_off7 = 0,
             src_off6 = 0, src_off5 = 0, src_off4 = 0;
    uint64_t dst_off3, src_off3;
    uint64_t dst_off, src_off; 
    uint64_t dst_stride3, src_stride3;

    if(dst->num_dims == 1) {
        BLOCK_SIZE_X = 256; 
        BLOCK_SIZE_Y = 4;
        BLOCK_SIZE_Z = 1;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = 1;
        sub_nz = 1;
        dst_off = dst->mat_view.lb[0];
        src_off = src->mat_view.lb[0];
    } else if(dst->num_dims == 2) {
        BLOCK_SIZE_X = 16;
        BLOCK_SIZE_Y = 32;
        BLOCK_SIZE_Z = 1;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = dst->mat_view.ub[1] - dst->mat_view.lb[1] + 1;
        sub_nz = 1;
        dst_off = dst->mat_view.lb[1] * dst->dist[0] + dst->mat_view.lb[0];
        src_off = src->mat_view.lb[1] * src->dist[0] + src->mat_view.lb[0];
    } else { 
        // ndims >= 3 will use 3D kernel in loops, so the params are the same
        BLOCK_SIZE_X = 8;
        BLOCK_SIZE_Y = 8;
        BLOCK_SIZE_Z = 16;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = dst->mat_view.ub[1] - dst->mat_view.lb[1] + 1;
        sub_nz = dst->mat_view.ub[2] - dst->mat_view.lb[2] + 1;

        dst_off3 = (dst->mat_view.lb[2] * dst->dist[1] + dst->mat_view.lb[1]) * dst->dist[0] + dst->mat_view.lb[0];
        src_off3 = (src->mat_view.lb[2] * src->dist[1] + src->mat_view.lb[1]) * src->dist[0] + src->mat_view.lb[0];

        dst_stride3 = dst->dist[2] * dst->dist[1] * dst->dist[0];
        src_stride3 = src->dist[2] * dst->dist[1] * dst->dist[0];

        // only ndims == 3 use fixed dst & src offset, others will change the values as excuted in loops
        if(dst->num_dims ==3) {
            dst_off = dst_off3;
            src_off = src_off3;
        }
    }

    GRID_SIZE_X = (sub_nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    GRID_SIZE_Y = (sub_ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    GRID_SIZE_Z = (sub_nz + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z;
    
    dim3 dimblock( BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z );
    dim3 dimgrid( GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z );

    switch(dst->num_dims) {
        case 1:
            goto ndimleq3;
            break;
        case 2:
            goto ndimleq3;
            break;
        case 3:
            goto ndimleq3;
            break;
        case 4:
            goto ndim4;
            break;
        case 5:
            goto ndim5;
            break;
        case 6:
            goto ndim6;
            break;
        case 7:
            goto ndim7;
            break;
        case 8:
            goto ndim8;
            break;
        case 9:
            goto ndim9;
            break;
        case 10:
            goto ndim10;
            break;
        default:
            return dspaces_ERR_INVALID_ARG;
            break;
        }

ndim10:
    for(dst9 = dst->mat_view.lb[9], src9 = src->mat_view.lb[9];
        dst9 <= dst->mat_view.ub[9]; dst9++, src9++) {
        dst_off9 = dst9 * dst->dist[8];
        src_off9 = src9 * src->dist[8];
    ndim9:
        for(dst8 = dst->mat_view.lb[8], src8 = src->mat_view.lb[8];
            dst8 <= dst->mat_view.ub[8]; dst8++, src8++) {
            dst_off8 = (dst_off9 + dst8) * dst->dist[7];
            src_off8 = (src_off9 + src8) * src->dist[7];
        ndim8:
            for(dst7 = dst->mat_view.lb[7], src7 = src->mat_view.lb[7];
                dst7 <= dst->mat_view.ub[7]; dst7++, src7++) {
                dst_off7 = (dst_off8 + dst7) * dst->dist[6];
                src_off7 = (src_off8 + src7) * src->dist[6];
            ndim7:
                for(dst6 = dst->mat_view.lb[6], src6 = src->mat_view.lb[6];
                    dst6 <= dst->mat_view.ub[6]; dst6++, src6++) {
                    dst_off6 = (dst_off7 + dst6) * dst->dist[5];
                    src_off6 = (src_off7 + src6) * src->dist[5];
                ndim6:
                    for(dst5 = dst->mat_view.lb[5], src5 = src->mat_view.lb[5];
                        dst5 <= dst->mat_view.ub[5]; dst5++, src5++) {
                        dst_off5 = (dst_off6 + dst5) * dst->dist[4];
                        src_off5 = (src_off6 + src5) * src->dist[4];
                    ndim5:
                        for(dst4 = dst->mat_view.lb[4], src4 = src->mat_view.lb[4];
                            dst4 <= dst->mat_view.ub[4]; dst4++, src4++) {
                            dst_off4 = (dst_off5 + dst4) * dst->dist[3];
                            src_off4 = (src_off5 + src4) * src->dist[3];
                        ndim4:
                            for(dst3 = dst->mat_view.lb[3], src3 = src->mat_view.lb[3];
                                dst3 <= dst->mat_view.ub[3]; dst3++, src3++) {
                                dst_off = (dst_off4 + dst3) * dst_stride3 + dst_off3;
                                src_off = (src_off4 + src3) * src_stride3 + dst_off3;
                            ndimleq3:
                                copy_subarray_f_double<<<dimgrid, dimblock, 0, *stream>>>(&d[dst_off],
                                    &s[src_off], dst->dist[0], dst->dist[1], dst->dist[2],
                                    src->dist[0], src->dist[1], src->dist[2], sub_nx, sub_ny, sub_nz);
                                if(src->num_dims <= 3)
                                    return dspaces_SUCCESS;
                            }
                            if(src->num_dims == 4)
                                return dspaces_SUCCESS;
                        }
                        if(src->num_dims == 5)
                            return dspaces_SUCCESS;
                    }
                    if(src->num_dims == 6)
                        return dspaces_SUCCESS;
                }
                if(src->num_dims == 7)
                    return dspaces_SUCCESS;
            }
            if(src->num_dims == 8)
                return dspaces_SUCCESS;
        }
        if(src->num_dims == 9)
            return dspaces_SUCCESS;
    }
    return dspaces_SUCCESS;
}

extern "C" int matrix_copy_cuda_f_float_async(struct matrix *dst, struct matrix *src, cudaStream_t *stream)
{
    float *d = (float*) dst->pdata;
    float *s = (float*) src->pdata;

    // int BLOCK_THREAD_SIZE = 1024;
    int BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z;
    int sub_nx, sub_ny, sub_nz;
    int GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z;

    uint64_t dst9, dst8, dst7, dst6, dst5, dst4, dst3;
    uint64_t dst_off9 = 0, dst_off8 = 0, dst_off7 = 0,
             dst_off6 = 0, dst_off5 = 0, dst_off4 = 0;
    uint64_t src9, src8, src7, src6, src5, src4, src3;
    uint64_t src_off9 = 0, src_off8 = 0, src_off7 = 0,
             src_off6 = 0, src_off5 = 0, src_off4 = 0;
    uint64_t dst_off3, src_off3;
    uint64_t dst_off, src_off; 
    uint64_t dst_stride3, src_stride3;

    if(dst->num_dims == 1) {
        BLOCK_SIZE_X = 256; 
        BLOCK_SIZE_Y = 4;
        BLOCK_SIZE_Z = 1;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = 1;
        sub_nz = 1;
        dst_off = dst->mat_view.lb[0];
        src_off = src->mat_view.lb[0];
    } else if(dst->num_dims == 2) {
        BLOCK_SIZE_X = 16;
        BLOCK_SIZE_Y = 32;
        BLOCK_SIZE_Z = 1;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = dst->mat_view.ub[1] - dst->mat_view.lb[1] + 1;
        sub_nz = 1;
        dst_off = dst->mat_view.lb[1] * dst->dist[0] + dst->mat_view.lb[0];
        src_off = src->mat_view.lb[1] * src->dist[0] + src->mat_view.lb[0];
    } else { 
        // ndims >= 3 will use 3D kernel in loops, so the params are the same
        BLOCK_SIZE_X = 8;
        BLOCK_SIZE_Y = 8;
        BLOCK_SIZE_Z = 16;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = dst->mat_view.ub[1] - dst->mat_view.lb[1] + 1;
        sub_nz = dst->mat_view.ub[2] - dst->mat_view.lb[2] + 1;

        dst_off3 = (dst->mat_view.lb[2] * dst->dist[1] + dst->mat_view.lb[1]) * dst->dist[0] + dst->mat_view.lb[0];
        src_off3 = (src->mat_view.lb[2] * src->dist[1] + src->mat_view.lb[1]) * src->dist[0] + src->mat_view.lb[0];

        dst_stride3 = dst->dist[2] * dst->dist[1] * dst->dist[0];
        src_stride3 = src->dist[2] * dst->dist[1] * dst->dist[0];

        // only ndims == 3 use fixed dst & src offset, others will change the values as excuted in loops
        if(dst->num_dims ==3) {
            dst_off = dst_off3;
            src_off = src_off3;
        }
    }

    GRID_SIZE_X = (sub_nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    GRID_SIZE_Y = (sub_ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    GRID_SIZE_Z = (sub_nz + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z;
    
    dim3 dimblock( BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z );
    dim3 dimgrid( GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z );

    switch(dst->num_dims) {
        case 1:
            goto ndimleq3;
            break;
        case 2:
            goto ndimleq3;
            break;
        case 3:
            goto ndimleq3;
            break;
        case 4:
            goto ndim4;
            break;
        case 5:
            goto ndim5;
            break;
        case 6:
            goto ndim6;
            break;
        case 7:
            goto ndim7;
            break;
        case 8:
            goto ndim8;
            break;
        case 9:
            goto ndim9;
            break;
        case 10:
            goto ndim10;
            break;
        default:
            return dspaces_ERR_INVALID_ARG;
            break;
        }

ndim10:
    for(dst9 = dst->mat_view.lb[9], src9 = src->mat_view.lb[9];
        dst9 <= dst->mat_view.ub[9]; dst9++, src9++) {
        dst_off9 = dst9 * dst->dist[8];
        src_off9 = src9 * src->dist[8];
    ndim9:
        for(dst8 = dst->mat_view.lb[8], src8 = src->mat_view.lb[8];
            dst8 <= dst->mat_view.ub[8]; dst8++, src8++) {
            dst_off8 = (dst_off9 + dst8) * dst->dist[7];
            src_off8 = (src_off9 + src8) * src->dist[7];
        ndim8:
            for(dst7 = dst->mat_view.lb[7], src7 = src->mat_view.lb[7];
                dst7 <= dst->mat_view.ub[7]; dst7++, src7++) {
                dst_off7 = (dst_off8 + dst7) * dst->dist[6];
                src_off7 = (src_off8 + src7) * src->dist[6];
            ndim7:
                for(dst6 = dst->mat_view.lb[6], src6 = src->mat_view.lb[6];
                    dst6 <= dst->mat_view.ub[6]; dst6++, src6++) {
                    dst_off6 = (dst_off7 + dst6) * dst->dist[5];
                    src_off6 = (src_off7 + src6) * src->dist[5];
                ndim6:
                    for(dst5 = dst->mat_view.lb[5], src5 = src->mat_view.lb[5];
                        dst5 <= dst->mat_view.ub[5]; dst5++, src5++) {
                        dst_off5 = (dst_off6 + dst5) * dst->dist[4];
                        src_off5 = (src_off6 + src5) * src->dist[4];
                    ndim5:
                        for(dst4 = dst->mat_view.lb[4], src4 = src->mat_view.lb[4];
                            dst4 <= dst->mat_view.ub[4]; dst4++, src4++) {
                            dst_off4 = (dst_off5 + dst4) * dst->dist[3];
                            src_off4 = (src_off5 + src4) * src->dist[3];
                        ndim4:
                            for(dst3 = dst->mat_view.lb[3], src3 = src->mat_view.lb[3];
                                dst3 <= dst->mat_view.ub[3]; dst3++, src3++) {
                                dst_off = (dst_off4 + dst3) * dst_stride3 + dst_off3;
                                src_off = (src_off4 + src3) * src_stride3 + dst_off3;
                            ndimleq3:
                                copy_subarray_f_float<<<dimgrid, dimblock, 0, *stream>>>(&d[dst_off],
                                    &s[src_off], dst->dist[0], dst->dist[1], dst->dist[2],
                                    src->dist[0], src->dist[1], src->dist[2], sub_nx, sub_ny, sub_nz);
                                if(src->num_dims <= 3)
                                    return dspaces_SUCCESS;
                            }
                            if(src->num_dims == 4)
                                return dspaces_SUCCESS;
                        }
                        if(src->num_dims == 5)
                            return dspaces_SUCCESS;
                    }
                    if(src->num_dims == 6)
                        return dspaces_SUCCESS;
                }
                if(src->num_dims == 7)
                    return dspaces_SUCCESS;
            }
            if(src->num_dims == 8)
                return dspaces_SUCCESS;
        }
        if(src->num_dims == 9)
            return dspaces_SUCCESS;
    }
    return dspaces_SUCCESS;
}

extern "C" int matrix_copy_cuda_f_short_async(struct matrix *dst, struct matrix *src, cudaStream_t *stream)
{
    short *d = (short*) dst->pdata;
    short *s = (short*) src->pdata;

    // int BLOCK_THREAD_SIZE = 1024;
    int BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z;
    int sub_nx, sub_ny, sub_nz;
    int GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z;

    uint64_t dst9, dst8, dst7, dst6, dst5, dst4, dst3;
    uint64_t dst_off9 = 0, dst_off8 = 0, dst_off7 = 0,
             dst_off6 = 0, dst_off5 = 0, dst_off4 = 0;
    uint64_t src9, src8, src7, src6, src5, src4, src3;
    uint64_t src_off9 = 0, src_off8 = 0, src_off7 = 0,
             src_off6 = 0, src_off5 = 0, src_off4 = 0;
    uint64_t dst_off3, src_off3;
    uint64_t dst_off, src_off; 
    uint64_t dst_stride3, src_stride3;

    if(dst->num_dims == 1) {
        BLOCK_SIZE_X = 256; 
        BLOCK_SIZE_Y = 4;
        BLOCK_SIZE_Z = 1;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = 1;
        sub_nz = 1;
        dst_off = dst->mat_view.lb[0];
        src_off = src->mat_view.lb[0];
    } else if(dst->num_dims == 2) {
        BLOCK_SIZE_X = 16;
        BLOCK_SIZE_Y = 32;
        BLOCK_SIZE_Z = 1;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = dst->mat_view.ub[1] - dst->mat_view.lb[1] + 1;
        sub_nz = 1;
        dst_off = dst->mat_view.lb[1] * dst->dist[0] + dst->mat_view.lb[0];
        src_off = src->mat_view.lb[1] * src->dist[0] + src->mat_view.lb[0];
    } else { 
        // ndims >= 3 will use 3D kernel in loops, so the params are the same
        BLOCK_SIZE_X = 8;
        BLOCK_SIZE_Y = 8;
        BLOCK_SIZE_Z = 16;
        sub_nx = dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1;
        sub_ny = dst->mat_view.ub[1] - dst->mat_view.lb[1] + 1;
        sub_nz = dst->mat_view.ub[2] - dst->mat_view.lb[2] + 1;

        dst_off3 = (dst->mat_view.lb[2] * dst->dist[1] + dst->mat_view.lb[1]) * dst->dist[0] + dst->mat_view.lb[0];
        src_off3 = (src->mat_view.lb[2] * src->dist[1] + src->mat_view.lb[1]) * src->dist[0] + src->mat_view.lb[0];

        dst_stride3 = dst->dist[2] * dst->dist[1] * dst->dist[0];
        src_stride3 = src->dist[2] * dst->dist[1] * dst->dist[0];

        // only ndims == 3 use fixed dst & src offset, others will change the values as excuted in loops
        if(dst->num_dims ==3) {
            dst_off = dst_off3;
            src_off = src_off3;
        }
    }

    GRID_SIZE_X = (sub_nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    GRID_SIZE_Y = (sub_ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    GRID_SIZE_Z = (sub_nz + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z;
    
    dim3 dimblock( BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z );
    dim3 dimgrid( GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z );

    switch(dst->num_dims) {
        case 1:
            goto ndimleq3;
            break;
        case 2:
            goto ndimleq3;
            break;
        case 3:
            goto ndimleq3;
            break;
        case 4:
            goto ndim4;
            break;
        case 5:
            goto ndim5;
            break;
        case 6:
            goto ndim6;
            break;
        case 7:
            goto ndim7;
            break;
        case 8:
            goto ndim8;
            break;
        case 9:
            goto ndim9;
            break;
        case 10:
            goto ndim10;
            break;
        default:
            return dspaces_ERR_INVALID_ARG;
            break;
        }

ndim10:
    for(dst9 = dst->mat_view.lb[9], src9 = src->mat_view.lb[9];
        dst9 <= dst->mat_view.ub[9]; dst9++, src9++) {
        dst_off9 = dst9 * dst->dist[8];
        src_off9 = src9 * src->dist[8];
    ndim9:
        for(dst8 = dst->mat_view.lb[8], src8 = src->mat_view.lb[8];
            dst8 <= dst->mat_view.ub[8]; dst8++, src8++) {
            dst_off8 = (dst_off9 + dst8) * dst->dist[7];
            src_off8 = (src_off9 + src8) * src->dist[7];
        ndim8:
            for(dst7 = dst->mat_view.lb[7], src7 = src->mat_view.lb[7];
                dst7 <= dst->mat_view.ub[7]; dst7++, src7++) {
                dst_off7 = (dst_off8 + dst7) * dst->dist[6];
                src_off7 = (src_off8 + src7) * src->dist[6];
            ndim7:
                for(dst6 = dst->mat_view.lb[6], src6 = src->mat_view.lb[6];
                    dst6 <= dst->mat_view.ub[6]; dst6++, src6++) {
                    dst_off6 = (dst_off7 + dst6) * dst->dist[5];
                    src_off6 = (src_off7 + src6) * src->dist[5];
                ndim6:
                    for(dst5 = dst->mat_view.lb[5], src5 = src->mat_view.lb[5];
                        dst5 <= dst->mat_view.ub[5]; dst5++, src5++) {
                        dst_off5 = (dst_off6 + dst5) * dst->dist[4];
                        src_off5 = (src_off6 + src5) * src->dist[4];
                    ndim5:
                        for(dst4 = dst->mat_view.lb[4], src4 = src->mat_view.lb[4];
                            dst4 <= dst->mat_view.ub[4]; dst4++, src4++) {
                            dst_off4 = (dst_off5 + dst4) * dst->dist[3];
                            src_off4 = (src_off5 + src4) * src->dist[3];
                        ndim4:
                            for(dst3 = dst->mat_view.lb[3], src3 = src->mat_view.lb[3];
                                dst3 <= dst->mat_view.ub[3]; dst3++, src3++) {
                                dst_off = (dst_off4 + dst3) * dst_stride3 + dst_off3;
                                src_off = (src_off4 + src3) * src_stride3 + dst_off3;
                            ndimleq3:
                                copy_subarray_f_short<<<dimgrid, dimblock, 0, *stream>>>(&d[dst_off],
                                    &s[src_off], dst->dist[0], dst->dist[1], dst->dist[2],
                                    src->dist[0], src->dist[1], src->dist[2], sub_nx, sub_ny, sub_nz);
                                if(src->num_dims <= 3)
                                    return dspaces_SUCCESS;
                            }
                            if(src->num_dims == 4)
                                return dspaces_SUCCESS;
                        }
                        if(src->num_dims == 5)
                            return dspaces_SUCCESS;
                    }
                    if(src->num_dims == 6)
                        return dspaces_SUCCESS;
                }
                if(src->num_dims == 7)
                    return dspaces_SUCCESS;
            }
            if(src->num_dims == 8)
                return dspaces_SUCCESS;
        }
        if(src->num_dims == 9)
            return dspaces_SUCCESS;
    }
    return dspaces_SUCCESS;
}

extern "C" int matrix_copy_cuda_f_char_async(struct matrix *dst, struct matrix *src, cudaStream_t *stream)
{
    char *d = (char*) dst->pdata;
    char *s = (char*) src->pdata;
    
    // int BLOCK_THREAD_SIZE = 1024;
    int BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z;
    int sub_nx, sub_ny, sub_nz;
    int GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z;

    uint64_t dst9, dst8, dst7, dst6, dst5, dst4, dst3;
    uint64_t dst_off9 = 0, dst_off8 = 0, dst_off7 = 0,
             dst_off6 = 0, dst_off5 = 0, dst_off4 = 0;
    uint64_t src9, src8, src7, src6, src5, src4, src3;
    uint64_t src_off9 = 0, src_off8 = 0, src_off7 = 0,
             src_off6 = 0, src_off5 = 0, src_off4 = 0;
    uint64_t dst_off3, src_off3;
    uint64_t dst_off, src_off; 
    uint64_t dst_stride3, src_stride3;

    // char function is used for arbitrary data types
    // Therefore, it needs to multiply elem_size to calculate the offsets and copy sizes
    if(dst->num_dims == 1) {
        BLOCK_SIZE_X = 256; 
        BLOCK_SIZE_Y = 4;
        BLOCK_SIZE_Z = 1;
        sub_nx = (dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1) * dst->size_elem;
        sub_ny = 1 * dst->size_elem;
        sub_nz = 1 * dst->size_elem;
        dst_off = dst->mat_view.lb[0] * dst->size_elem;
        src_off = src->mat_view.lb[0] * dst->size_elem;
    } else if(dst->num_dims == 2) {
        BLOCK_SIZE_X = 16;
        BLOCK_SIZE_Y = 32;
        BLOCK_SIZE_Z = 1;
        sub_nx = (dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1) * dst->size_elem;
        sub_ny = (dst->mat_view.ub[1] - dst->mat_view.lb[1] + 1) * dst->size_elem;
        sub_nz = 1 * dst->size_elem;
        dst_off = (dst->mat_view.lb[1] * dst->dist[0] + dst->mat_view.lb[0]) * dst->size_elem;
        src_off = (src->mat_view.lb[1] * src->dist[0] + src->mat_view.lb[0]) * dst->size_elem;
    } else { 
        // ndims >= 3 will use 3D kernel in loops, so the params are the same
        BLOCK_SIZE_X = 8;
        BLOCK_SIZE_Y = 8;
        BLOCK_SIZE_Z = 16;
        sub_nx = (dst->mat_view.ub[0] - dst->mat_view.lb[0] + 1) * dst->size_elem;
        sub_ny = (dst->mat_view.ub[1] - dst->mat_view.lb[1] + 1) * dst->size_elem;
        sub_nz = (dst->mat_view.ub[2] - dst->mat_view.lb[2] + 1) * dst->size_elem;

        dst_off3 = ((dst->mat_view.lb[2] * dst->dist[1] + dst->mat_view.lb[1]) * dst->dist[0] + dst->mat_view.lb[0]) * dst->size_elem;
        src_off3 = ((src->mat_view.lb[2] * src->dist[1] + src->mat_view.lb[1]) * src->dist[0] + src->mat_view.lb[0]) * dst->size_elem;

        dst_stride3 = (dst->dist[2] * dst->dist[1] * dst->dist[0]) * dst->size_elem;
        src_stride3 = (src->dist[2] * dst->dist[1] * dst->dist[0]) * dst->size_elem;

        // only ndims == 3 use fixed dst & src offset, others will change the values as excuted in loops
        if(dst->num_dims ==3) {
            dst_off = dst_off3;
            src_off = src_off3;
        }
    }

    GRID_SIZE_X = (sub_nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    GRID_SIZE_Y = (sub_ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    GRID_SIZE_Z = (sub_nz + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z;
    
    dim3 dimblock( BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z );
    dim3 dimgrid( GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z );

    switch(dst->num_dims) {
        case 1:
            goto ndimleq3;
            break;
        case 2:
            goto ndimleq3;
            break;
        case 3:
            goto ndimleq3;
            break;
        case 4:
            goto ndim4;
            break;
        case 5:
            goto ndim5;
            break;
        case 6:
            goto ndim6;
            break;
        case 7:
            goto ndim7;
            break;
        case 8:
            goto ndim8;
            break;
        case 9:
            goto ndim9;
            break;
        case 10:
            goto ndim10;
            break;
        default:
            return dspaces_ERR_INVALID_ARG;
            break;
        }

ndim10:
    for(dst9 = dst->mat_view.lb[9], src9 = src->mat_view.lb[9];
        dst9 <= dst->mat_view.ub[9]; dst9++, src9++) {
        dst_off9 = dst9 * dst->dist[8];
        src_off9 = src9 * src->dist[8];
    ndim9:
        for(dst8 = dst->mat_view.lb[8], src8 = src->mat_view.lb[8];
            dst8 <= dst->mat_view.ub[8]; dst8++, src8++) {
            dst_off8 = (dst_off9 + dst8) * dst->dist[7];
            src_off8 = (src_off9 + src8) * src->dist[7];
        ndim8:
            for(dst7 = dst->mat_view.lb[7], src7 = src->mat_view.lb[7];
                dst7 <= dst->mat_view.ub[7]; dst7++, src7++) {
                dst_off7 = (dst_off8 + dst7) * dst->dist[6];
                src_off7 = (src_off8 + src7) * src->dist[6];
            ndim7:
                for(dst6 = dst->mat_view.lb[6], src6 = src->mat_view.lb[6];
                    dst6 <= dst->mat_view.ub[6]; dst6++, src6++) {
                    dst_off6 = (dst_off7 + dst6) * dst->dist[5];
                    src_off6 = (src_off7 + src6) * src->dist[5];
                ndim6:
                    for(dst5 = dst->mat_view.lb[5], src5 = src->mat_view.lb[5];
                        dst5 <= dst->mat_view.ub[5]; dst5++, src5++) {
                        dst_off5 = (dst_off6 + dst5) * dst->dist[4];
                        src_off5 = (src_off6 + src5) * src->dist[4];
                    ndim5:
                        for(dst4 = dst->mat_view.lb[4], src4 = src->mat_view.lb[4];
                            dst4 <= dst->mat_view.ub[4]; dst4++, src4++) {
                            dst_off4 = (dst_off5 + dst4) * dst->dist[3];
                            src_off4 = (src_off5 + src4) * src->dist[3];
                        ndim4:
                            for(dst3 = dst->mat_view.lb[3], src3 = src->mat_view.lb[3];
                                dst3 <= dst->mat_view.ub[3]; dst3++, src3++) {
                                dst_off = (dst_off4 + dst3) * dst_stride3 + dst_off3;
                                src_off = (src_off4 + src3) * src_stride3 + dst_off3;
                            ndimleq3:
                                copy_subarray_f_char<<<dimgrid, dimblock, 0, *stream>>>(&d[dst_off],
                                    &s[src_off], dst->dist[0], dst->dist[1], dst->dist[2],
                                    src->dist[0], src->dist[1], src->dist[2], sub_nx, sub_ny, sub_nz);
                                if(src->num_dims <= 3)
                                    return dspaces_SUCCESS;
                            }
                            if(src->num_dims == 4)
                                return dspaces_SUCCESS;
                        }
                        if(src->num_dims == 5)
                            return dspaces_SUCCESS;
                    }
                    if(src->num_dims == 6)
                        return dspaces_SUCCESS;
                }
                if(src->num_dims == 7)
                    return dspaces_SUCCESS;
            }
            if(src->num_dims == 8)
                return dspaces_SUCCESS;
        }
        if(src->num_dims == 9)
            return dspaces_SUCCESS;
    }
    return dspaces_SUCCESS;
}
