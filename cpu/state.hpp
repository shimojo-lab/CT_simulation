#include <iostream>
#include "type.hpp"

#define sqrt2 1.41421356237309504880

typedef unsigned int UINT;
typedef ZOmega CTYPE;     // Complex型

class QuantumState
{
    UINT dim;
    CTYPE *vec;
    UINT k;

public:
    QuantumState(UINT n);
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
}

std::ostream& operator<<(std::ostream& os, const QuantumState& x)
{
    ZOmega tmp = x.vec[0];
    for(ITYPE i = 0; i < x.dim; i++){
        ZOmega value = x.vec[i];
        os << convert(x.vec[i]) / std::pow(sqrt2, (double)x.k) << "    " << value << "\n";
    }
    os << "k = " << x.k;
    //os << x.vec << "\nk = " << x.k;
    return os;
}

void QuantumState::act_H(UINT target)
{
    const UINT loop = dim >> 1;
    const UINT mask = (1 << target);
    const UINT mask_low = mask - 1;
    const UINT mask_high = ~mask_low;

    for(UINT i = 0; i < loop; i++){
        UINT zero_idx = (i & mask_low) | ((i & mask_high) << 1);
        UINT one_idx = zero_idx + mask;

        CTYPE tmp = vec[zero_idx] + vec[one_idx];
        vec[one_idx] = vec[zero_idx] - vec[one_idx];
        vec[zero_idx] = tmp;
    }
    k++;
}

void QuantumState::act_T(UINT target)
{
    const UINT loop = dim >> 1;
    const UINT mask = (1 << target);
    const UINT mask_low = mask - 1;
    const UINT mask_high = ~mask_low;

    for(UINT i = 0; i < loop; i++){
        UINT one_idx = (i & mask_low) | ((i & mask_high) << 1) | mask;
        vec[one_idx] = multiple_omega(vec[one_idx]);
    }
}

void QuantumState::act_S(UINT target)
{
    const UINT loop = dim >> 1;
    const UINT mask = (1 << target);
    const UINT mask_low = mask - 1;
    const UINT mask_high = ~mask_low;

    for(UINT i = 0; i < loop; i++){
        UINT one_idx = (i & mask_low) | ((i & mask_high) << 1) | mask;
        vec[one_idx] = multiple_i(vec[one_idx]);
    }
}

void QuantumState::act_CNOT(UINT control, UINT target)
{
    const UINT loop = dim >> 2;

    const UINT mask_control = (1 << control);
    const UINT mask_low_control = mask_control - 1;
    const UINT mask_high_control = ~mask_low_control;

    const UINT mask_target = (1 << target);
    const UINT mask_low_target = mask_target - 1;
    const UINT mask_high_target = ~mask_low_target;

    if(control < target){
        for(UINT i = 0; i < loop; i++){
            UINT one_control_idx = (i & mask_low_control) | ((i & mask_high_control) << 1) | mask_control;
            UINT one_zero_idx = (one_control_idx & mask_low_target) | ((one_control_idx & mask_high_target) << 1);
            UINT one_one_idx = one_zero_idx | mask_target;

            CTYPE tmp = vec[one_zero_idx];
            vec[one_zero_idx] = vec[one_one_idx];
            vec[one_one_idx] = tmp;
        }
    }else{
        for(UINT i = 0; i < loop; i++){
            UINT one_target_idx = (i & mask_low_target) | ((i & mask_high_target) << 1) | mask_target;
            UINT one_zero_idx = (one_target_idx & mask_low_control) | ((one_target_idx & mask_high_control) << 1);
            UINT one_one_idx = one_zero_idx | mask_control;   

            CTYPE tmp = vec[one_zero_idx];
            vec[one_zero_idx] = vec[one_one_idx];
            vec[one_one_idx] = tmp;
        }
    }
}

void QuantumState::act_1qubits_gate(UINT target,  CTYPE *mat)
{
    const UINT loop = dim >> 1;
    const UINT mask = (1 << target);
    const UINT mask_low = mask - 1;
    const UINT mask_high = ~mask_low;

    for(UINT i = 0; i < dim>>1; i++){
        UINT zero_idx = (i & mask_low) | ((i & mask_high) << 1);
        UINT one_idx = zero_idx | mask;

        CTYPE tmp = mat[0]*vec[zero_idx] + mat[1]*vec[one_idx];
        vec[one_idx] = mat[2]*vec[zero_idx] - mat[3]*vec[one_idx];
        vec[zero_idx] = tmp;
    }
}