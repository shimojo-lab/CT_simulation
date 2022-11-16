#include <iostream>
#include "type.hpp"

typedef unsigned int UINT;
typedef ZOmega CTYPE;     // Complex型

class QuantumState
{
    UINT dim;
    CTYPE *vec;
    CTYPE *vec_gpu;
    UINT k;

public:
    QuantumState(UINT n);
    CTYPE* get_vec();
    friend std::ostream& operator<<(std::ostream& os, const QuantumState& x);
    void act_H(UINT target);
    void act_S(UINT target);
    void act_T(UINT target);
    void act_CNOT(UINT control, UINT target);
    void act_1qubits_gate(UINT target, CTYPE *mat);
};

QuantumState::QuantumState(UINT n){
    k = 0;
    dim = 1 << n;
    vec = new CTYPE[dim];
    vec[0] = make_ZOmega(0,0,0,1);

    cudaError_t err;
    err = cudaMalloc((void**)&vec_gpu, sizeof(CTYPE)*dim);
    if (err != cudaSuccess) {
        printf("cudaMallocエラー\n");
        exit(err);
    }

    cudaMemcpy(vec_gpu, vec, sizeof(CTYPE)*dim, cudaMemcpyHostToDevice);
}

CTYPE* QuantumState::get_vec()
{
    cudaMemcpy(vec, vec_gpu, sizeof(CTYPE)*dim, cudaMemcpyDeviceToHost);
    return vec;
}

std::ostream& operator<<(std::ostream& os, const QuantumState& x)
{
    for(UINT i = 0; i < x.dim; i++){
        const double sqrt2 = sqrt(2.0);
        os << convert(x.vec[i]) / std::pow(sqrt2, (double)x.k) << "    " << value << "\n";
    }
    os << "k = " << x.k;
    //os << x.vec << "\nk = " << x.k;
    return os;
}

__global__ void act_H_gpu(CTYPE *vec, UINT target, UINT dim)
{
    UINT j = blockIdx.x * blockDim.x + threadIdx.x;
    UINT zero_idx, one_idx;
    CTYPE tmp;

    if ((j < (dim >> 1))) {
        zero_idx = (j >> target);
        zero_idx = zero_idx << (target + 1);
        zero_idx += j & ((1 << target) - 1);
        one_idx = zero_idx ^ (1 << target);

        tmp = vec[zero_idx];
        vec[zero_idx] = tmp + vec[one_idx];
        vec[one_idx] = tmp - vec[one_idx];

        //vec[zero_idx] = make_ZOmega(vec[zero_idx].x*3, vec[zero_idx].y*3, vec[zero_idx].z*3, vec[zero_idx].w*3);
        //vec[one_idx] = make_ZOmega(vec[one_idx].x*3, vec[one_idx].y*3, vec[one_idx].z*3, vec[one_idx].w*3);
    }
}

void QuantumState::act_H(UINT target)
{
    UINT half_dim = dim >> 1;
    UINT block = half_dim <= 1024 ? half_dim : 1024;
    UINT grid = half_dim / block;

    act_H_gpu <<<grid, block>>> (vec_gpu, target, dim);
    k++;
}

__global__ void act_T_gpu(CTYPE *vec, UINT target, UINT dim)
{
    UINT j = blockIdx.x * blockDim.x + threadIdx.x;
    UINT zero_idx, one_idx;

    if ((j < (dim >> 1))) {
        zero_idx = (j >> target);
        zero_idx = zero_idx << (target + 1);
        zero_idx |= j & ((1 << target) - 1);
        one_idx = zero_idx ^ (1 << target);

        vec[one_idx] = make_ZOmega(vec[one_idx].y, vec[one_idx].z, vec[one_idx].w, -vec[one_idx].x);
    }
}

void QuantumState::act_T(UINT target)
{
    UINT half_dim = dim >> 1;
    UINT block = half_dim <= 1024 ? half_dim : 1024;
    UINT grid = half_dim / block;

    act_T_gpu <<<grid, block>>> (vec_gpu, target, dim);
}

__global__ void act_S_gpu(CTYPE *vec, UINT target, UINT dim)
{
    UINT j = blockIdx.x * blockDim.x + threadIdx.x;
    UINT zero_idx, one_idx;

    if ((j < (dim >> 1))) {
        zero_idx = (j >> target);
        zero_idx = zero_idx << (target + 1);
        zero_idx += j & ((1 << target) - 1);
        one_idx = zero_idx ^ (1 << target);

        vec[one_idx] = make_ZOmega(-vec[one_idx].z, -vec[one_idx].w, vec[one_idx].x, vec[one_idx].y);
    }
}

void QuantumState::act_S(UINT target)
{
    UINT half_dim = dim >> 1;
    UINT block = half_dim <= 1024 ? half_dim : 1024;
    UINT grid = half_dim / block;

    act_S_gpu <<<grid, block>>> (vec_gpu, target, dim);
}


// control < targetの場合
__global__ void act_CNOT_gpu1(CTYPE *vec, UINT target, UINT control, UINT dim)
{
    UINT j = blockIdx.x * blockDim.x + threadIdx.x;
    UINT one_control_idx, one_zero_idx, one_one_idx;
    CTYPE tmp;

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
__global__ void act_CNOT_gpu2(CTYPE *vec, UINT target, UINT control, UINT dim)
{
    UINT j = blockIdx.x * blockDim.x + threadIdx.x;
    UINT one_target_idx, one_zero_idx, one_one_idx;
    CTYPE tmp;

    if ((j < (dim >> 2))) {
        one_target_idx = (((j >> target) << (target + 1)) | (j & ((1 << target) -1))) | (1 << target);
        one_zero_idx = ((one_target_idx >> control) << (control + 1)) | (one_target_idx & ((1 << control) -1));
        one_one_idx = one_zero_idx | (1 << control);

        tmp = vec[one_zero_idx];
        vec[one_zero_idx] = vec[one_one_idx];
        vec[one_one_idx] = tmp;
    }
}

void QuantumState::act_CNOT(UINT control, UINT target)
{
    UINT quarter_dim = dim >> 2;
    UINT block = quarter_dim <= 1024 ? quarter_dim : 1024;
    UINT grid = quarter_dim / block;

    if(control < target) act_CNOT_gpu1 <<<grid, block>>> (vec_gpu, target, control, dim);
    else act_CNOT_gpu2 <<<grid, block>>> (vec_gpu, target, control, dim);
}

// void QuantumState::act_1qubits_gate(UINT target,  CTYPE *mat)
// {
//     const UINT loop = dim >> 1;
//     const UINT mask = (1 << target);
//     const UINT mask_low = mask - 1;
//     const UINT mask_high = ~mask_low;

//     for(UINT i = 0; i < dim>>1; i++){
//         UINT zero_idx = (i & mask_low) | ((i & mask_high) << 1);
//         UINT one_idx = zero_idx | mask;

//         CTYPE tmp = mat[0]*vec[zero_idx] + mat[1]*vec[one_idx];
//         vec[one_idx] = mat[2]*vec[zero_idx] - mat[3]*vec[one_idx];
//         vec[zero_idx] = tmp;
//     }
// }
