#include <iostream>
#include <complex>

typedef unsigned int UINT;
typedef std::complex<double> exist_CTYPE;     // Complex型

class exist_QuantumState
{
    UINT dim;
    exist_CTYPE *vec;

public:
    exist_QuantumState(UINT n);
    exist_CTYPE* get_vec() {return vec;}
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
    vec[0] = exist_CTYPE(1.0, 0.0);
}

std::ostream& operator<<(std::ostream& os, const exist_QuantumState& x)
{
    for(UINT i = 0; i < x.dim; i++){
        os << x.vec[i] << "\n";
    }
    return os;
}

void exist_QuantumState::act_H(UINT target)
{
    const UINT loop = dim >> 1;
    const UINT mask = (1 << target);
    const UINT mask_low = mask - 1;
    const UINT mask_high = ~mask_low;
    const double sqrt2_inv = 1 / sqrt(2.0);

    for(UINT i = 0; i < loop; i++){
        UINT zero_idx = (i & mask_low) | ((i & mask_high) << 1);
        UINT one_idx = zero_idx + mask;

        exist_CTYPE tmp = vec[zero_idx];
        vec[zero_idx] = (tmp + vec[one_idx]) * sqrt2_inv;
        vec[zero_idx] = (tmp - vec[one_idx]) * sqrt2_inv;
    }
}

void exist_QuantumState::act_T(UINT target)
{
    const UINT loop = dim >> 1;
    const UINT mask = (1 << target);
    const UINT mask_low = mask - 1;
    const UINT mask_high = ~mask_low;
    const exist_CTYPE omega(1 / sqrt(2.0), 1 / sqrt(2.0));

    for(UINT i = 0; i < loop; i++){
        UINT one_idx = (i & mask_low) | ((i & mask_high) << 1) | mask;
        vec[one_idx] = omega * vec[one_idx];
    }
}

void exist_QuantumState::act_S(UINT target)
{
    const UINT loop = dim >> 1;
    const UINT mask = (1 << target);
    const UINT mask_low = mask - 1;
    const UINT mask_high = ~mask_low;
    // const exist_CTYPE imag_unit(0.0, 1.0); 

    for(UINT i = 0; i < loop; i++){
        UINT one_idx = (i & mask_low) | ((i & mask_high) << 1) | mask;
        // vec[one_idx] = imag_unit * vec[one_idx];
        vec[one_idx] = exist_CTYPE(-vec[one_idx].imag(), vec[one_idx].real());
    }
}

void exist_QuantumState::act_CNOT(UINT control, UINT target)
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

            exist_CTYPE tmp = vec[one_zero_idx];
            vec[one_zero_idx] = vec[one_one_idx];
            vec[one_one_idx] = tmp;
        }
    }else{
        for(UINT i = 0; i < loop; i++){
            UINT one_target_idx = (i & mask_low_target) | ((i & mask_high_target) << 1) | mask_target;
            UINT one_zero_idx = (one_target_idx & mask_low_control) | ((one_target_idx & mask_high_control) << 1);
            UINT one_one_idx = one_zero_idx | mask_control;   

            exist_CTYPE tmp = vec[one_zero_idx];
            vec[one_zero_idx] = vec[one_one_idx];
            vec[one_one_idx] = tmp;
        }
    }
}

void exist_QuantumState::act_1qubits_gate(UINT target,  exist_CTYPE *mat)
{
    const UINT loop = dim >> 1;
    const UINT mask = (1 << target);
    const UINT mask_low = mask - 1;
    const UINT mask_high = ~mask_low;

    for(UINT i = 0; i < dim>>1; i++){
        UINT zero_idx = (i & mask_low) | ((i & mask_high) << 1);
        UINT one_idx = zero_idx | mask;

        exist_CTYPE tmp = mat[0]*vec[zero_idx] + mat[1]*vec[one_idx];
        vec[one_idx] = mat[2]*vec[zero_idx] - mat[3]*vec[one_idx];
        vec[zero_idx] = tmp;
    }
}