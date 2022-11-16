#include <iostream>
#include <cuComplex.h>

typedef unsigned int UINT;
typedef cuDoubleComplex exist_CTYPE;     // Complex型

class exist_QuantumState
{
    UINT dim;
    exist_CTYPE *vec;
    exist_CTYPE *vec_gpu;

public:
    exist_QuantumState(UINT n);
    exist_CTYPE* get_vec();
    exist_CTYPE* get_vec_gpu() {return vec_gpu;}
    friend std::ostream& operator<<(std::ostream& os, const exist_QuantumState& x);
    void act_H(UINT target);
    void act_S(UINT target);
    void act_T(UINT target);
    void act_CNOT(UINT control, UINT target);
    void act_1qubits_gate(UINT target, exist_CTYPE *mat);
};

exist_QuantumState::exist_QuantumState(UINT n){
    dim = 1 << n;
    vec = new exist_CTYPE[dim];
    vec[0] = make_cuDoubleComplex(1., 0.);

    cudaError_t err;
    err = cudaMalloc((void**)&vec_gpu, sizeof(exist_CTYPE)*dim);
    if (err != cudaSuccess) {
        printf("cudaMallocエラー\n");
        exit(err);
    }

    cudaMemcpy(vec_gpu, vec, sizeof(exist_CTYPE)*dim, cudaMemcpyHostToDevice);
}

exist_CTYPE* exist_QuantumState::get_vec()
{
    cudaMemcpy(vec, vec_gpu, sizeof(exist_CTYPE)*dim, cudaMemcpyDeviceToHost);
    return vec;
}

std::ostream& operator<<(std::ostream& os, const exist_QuantumState& x)
{
    for(UINT i = 0; i < x.dim; i++){
        os << x.vec[i].x << ", " << x.vec[i].y << "\n";
    }
    return os;
}

__global__ void act_H_gpu(exist_CTYPE *vec, UINT target, UINT dim)
{
    UINT j = blockIdx.x * blockDim.x + threadIdx.x;
    UINT zero_idx, one_idx;
    exist_CTYPE tmp;
    double sqrt2_inv = 1.0 / sqrt(2.0);

    if ((j < (dim >> 1))) {
        zero_idx = (j >> target);
        zero_idx = zero_idx << (target + 1);
        zero_idx += j & ((1 << target) - 1);
        one_idx = zero_idx ^ (1 << target);

        //zero_idx = j;
        //one_idx = j + (dim >> 1);

        tmp = vec[zero_idx];
        vec[zero_idx] = cuCadd(tmp, vec[one_idx]);
        vec[one_idx] = cuCadd(tmp, make_cuDoubleComplex(-vec[one_idx].x, -vec[one_idx].y));
        
        vec[zero_idx]= make_cuDoubleComplex(vec[zero_idx].x * sqrt2_inv, vec[zero_idx].y * sqrt2_inv);
        vec[one_idx] = make_cuDoubleComplex(vec[one_idx].x * sqrt2_inv, vec[one_idx].x * sqrt2_inv);
    }
}

void exist_QuantumState::act_H(UINT target)
{
    UINT half_dim = dim >> 1;
    UINT block = half_dim <= 1024 ? half_dim : 1024;
    UINT grid = half_dim / block;

    act_H_gpu <<<grid, block>>> (vec_gpu, target, dim);
}

__global__ void act_T_gpu(exist_CTYPE *vec, UINT target, UINT dim)
{
    UINT j = blockIdx.x * blockDim.x + threadIdx.x;
    UINT zero_idx, one_idx;
    exist_CTYPE omega = make_cuDoubleComplex(1.0 / sqrt(2.0), 1.0 / sqrt(2.0));

    if ((j < (dim >> 1))) {
        zero_idx = (j >> target);
        zero_idx = zero_idx << (target + 1);
        zero_idx += j & ((1 << target) - 1);
        one_idx = zero_idx ^ (1 << target);

        vec[one_idx] = cuCmul(vec[one_idx], omega);
    }
}

void exist_QuantumState::act_T(UINT target)
{
    UINT half_dim = dim >> 1;
    UINT block = half_dim <= 1024 ? half_dim : 1024;
    UINT grid = half_dim / block;

    act_T_gpu <<<grid, block>>> (vec_gpu, target, dim);
}

__global__ void act_S_gpu(exist_CTYPE *vec, UINT target, UINT dim)
{
    UINT j = blockIdx.x * blockDim.x + threadIdx.x;
    UINT zero_idx, one_idx;

    if ((j < (dim >> 1))) {
        zero_idx = (j >> target);
        zero_idx = zero_idx << (target + 1);
        zero_idx += j & ((1 << target) - 1);
        one_idx = zero_idx ^ (1 << target);

        vec[one_idx] = make_cuDoubleComplex(-vec[one_idx].y, vec[one_idx].x);
    }
}

void exist_QuantumState::act_S(UINT target)
{
    UINT half_dim = dim >> 1;
    UINT block = half_dim <= 1024 ? half_dim : 1024;
    UINT grid = half_dim / block;

    act_S_gpu <<<grid, block>>> (vec_gpu, target, dim);
}


// control < targetの場合
__global__ void act_CNOT_gpu1(exist_CTYPE *vec, UINT target, UINT control, UINT dim)
{
    UINT j = blockIdx.x * blockDim.x + threadIdx.x;
    UINT one_control_idx, one_zero_idx, one_one_idx;
    exist_CTYPE tmp;

    if ((j < (dim >> 2))) {
        one_control_idx = (((j >> control) << (control + 1)) | (j & ((1 << control) -1))) | (1 << control);
        one_zero_idx = ((one_control_idx >> target) << (target + 1)) | (one_control_idx & ((1 << target) -1));
        one_one_idx = one_zero_idx | (1 << target);

        tmp = vec[one_zero_idx];
        vec[one_zero_idx] = vec[one_one_idx];
        vec[one_one_idx] = tmp;
    }
}

// control > targetの場合 
__global__ void act_CNOT_gpu2(exist_CTYPE *vec, UINT target, UINT control, UINT dim)
{
    UINT j = blockIdx.x * blockDim.x + threadIdx.x;
    UINT one_target_idx, one_zero_idx, one_one_idx;
    exist_CTYPE tmp;

    if ((j < (dim >> 2))) {
        one_target_idx = (((j >> target) << (target + 1)) | (j & ((1 << target) -1))) | (1 << target);
        one_zero_idx = ((one_target_idx >> control) << (control + 1)) | (one_target_idx & ((1 << control) -1));
        one_one_idx = one_zero_idx | (1 << control);

        tmp = vec[one_zero_idx];
        vec[one_zero_idx] = vec[one_one_idx];
        vec[one_one_idx] = tmp;
    }
}

void exist_QuantumState::act_CNOT(UINT control, UINT target)
{
    UINT quarter_dim = dim >> 2;
    UINT block = quarter_dim <= 1024 ? quarter_dim : 1024;
    UINT grid = quarter_dim / block;

    if(control < target) act_CNOT_gpu1 <<<grid, block>>> (vec_gpu, target, control, dim);
    else act_CNOT_gpu2 <<<grid, block>>> (vec_gpu, target, control, dim);
}

// void exist_QuantumState::act_1qubits_gate(UINT target,  exist_CTYPE *mat)
// {
//     const UINT loop = dim >> 1;
//     const UINT mask = (1 << target);
//     const UINT mask_low = mask - 1;
//     const UINT mask_high = ~mask_low;

//     for(UINT i = 0; i < dim>>1; i++){
//         UINT zero_idx = (i & mask_low) | ((i & mask_high) << 1);
//         UINT one_idx = zero_idx | mask;

//         exist_CTYPE tmp = mat[0]*vec[zero_idx] + mat[1]*vec[one_idx];
//         vec[one_idx] = mat[2]*vec[zero_idx] - mat[3]*vec[one_idx];
//         vec[zero_idx] = tmp;
//     }
// }
