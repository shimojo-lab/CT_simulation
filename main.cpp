#include <vector>
#include <complex>
#include <string>
#include <math.h>
#include <iostream>
#include <vector>
#include <Rings.hpp>

//typedef std::complex<ITYPE> state_type;
typedef ZOmega state_type;
typedef int ITYPE;

class QuantumState
{
    ITYPE dim;
    state_type *vec;
    ITYPE k;

public:
    QuantumState(ITYPE n){
        k = 0;
        dim = 1 << n;
        vec = new state_type[dim];
        vec[0] = ZOmega(0,0,0,1);
    }
    friend std::ostream& operator<<(std::ostream& os, const QuantumState& x);
    void act_H(ITYPE target);
    void act_S(ITYPE target);
    void act_T(ITYPE target);
    void act_CNOT(ITYPE control, ITYPE target);
    void act_1qubits_gate(ITYPE target, state_type *mat);
};

std::ostream& operator<<(std::ostream& os, const QuantumState& x)
{
    ZOmega tmp = x.vec[0];
    for(ITYPE i = 0; i < x.dim; i++){
        ZOmega value = x.vec[i];
        os << value.get_complex() / std::pow(Root2, (long double)x.k) << "    " << value << "\n";
    }
    os << "k = " << x.k;
    //os << x.vec << "\nk = " << x.k;
    return os;
}

void QuantumState::act_H(ITYPE target)
{
    const ITYPE loop = dim >> 1;
    const ITYPE mask = (1 << target);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    for(ITYPE i = 0; i < dim>>1; i++){
        ITYPE zero_idx = (i & mask_low) | ((i & mask_high) << 1);
        ITYPE one_idx = zero_idx + mask;

        state_type tmp = vec[zero_idx] + vec[one_idx];
        vec[one_idx] = vec[zero_idx] - vec[one_idx];
        vec[zero_idx] = tmp;
    }
    k++;
}

void QuantumState::act_T(ITYPE target)
{
    const ITYPE loop = dim >> 1;
    const ITYPE mask = (1 << target);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    for(ITYPE i = 0; i < loop; i++){
        ITYPE zero_idx = (i & mask_low) | ((i & mask_high) << 1) ;
        //ITYPE one_idx = (i & mask_low) | ((i & mask_high) << 1) | mask;
        ITYPE one_idx = zero_idx | mask;
        vec[one_idx].multiple_omega();
        vec[zero_idx].multiple_omega();
    }
}

void QuantumState::act_S(ITYPE target)
{
    const ITYPE loop = dim >> 1;
    const ITYPE mask = (1 << target);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    for(ITYPE i = 0; i < loop; i++){
        ITYPE one_idx = (i & mask_low) | ((i & mask_high) << 1) | mask;
        vec[one_idx].conj();
    }
}

void QuantumState::act_CNOT(ITYPE control, ITYPE target)
{
    const ITYPE loop = dim >> 2;

    const ITYPE mask_control = (1 << control);
    const ITYPE mask_low_control = mask_control - 1;
    const ITYPE mask_high_control = ~mask_low_control;

    const ITYPE mask_target = (1 << target);
    const ITYPE mask_low_target = mask_target - 1;
    const ITYPE mask_high_target = ~mask_low_target;

    for(ITYPE i = 0; i < dim>>2; i++){
        if(control < target){
            ITYPE one_control_idx = (i & mask_low_control) | ((i & mask_high_control) << 1) | mask_control;
            ITYPE one_zero_idx = (one_control_idx & mask_low_target) | ((one_control_idx & mask_high_target) << 1);
            ITYPE one_one_idx = one_zero_idx | mask_target;

            state_type tmp = vec[one_zero_idx];
            vec[one_zero_idx] = vec[one_one_idx];
            vec[one_one_idx] = tmp;
        }
        else{
            ITYPE one_target_idx = (i & mask_low_target) | ((i & mask_high_target) << 1) | mask_target;
            ITYPE one_zero_idx = (one_target_idx & mask_low_control) | ((one_target_idx & mask_high_control) << 1);
            ITYPE one_one_idx = one_zero_idx | mask_control;   

            state_type tmp = vec[one_zero_idx];
            vec[one_zero_idx] = vec[one_one_idx];
            vec[one_one_idx] = tmp;
        }
    }
}

void QuantumState::act_1qubits_gate(ITYPE target,  state_type *mat)
{
    const ITYPE loop = dim >> 1;
    const ITYPE mask = (1 << target);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    for(ITYPE i = 0; i < dim>>1; i++){
        ITYPE zero_idx = (i & mask_low) | ((i & mask_high) << 1);
        ITYPE one_idx = zero_idx | mask;

        state_type tmp = mat[0]*vec[zero_idx] + mat[1]*vec[one_idx];
        vec[one_idx] = mat[2]*vec[zero_idx] - mat[3]*vec[one_idx];
        vec[zero_idx] = tmp;
    }
}

int main()
{
    ITYPE n = 28;
    ITYPE loop = 100000;
    //loop = 1;
    //std::cin >> n;
    QuantumState state(n);
    clock_t start, end;
    double time;

    state.act_H(0);

    start = clock();
    state.act_H(10);
    end = clock();
    time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("H %lf[ms]\n", time);

    start = clock();
    state.act_S(10);
    end = clock();
    time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("S %lf[ms]\n", time);

    start = clock();
    state.act_T(10);
    end = clock();
    time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("T %lf[ms]\n", time);

    start = clock();
    state.act_CNOT(4,8);
    end = clock();
    time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("CNOT %lf[ms]\n", time);

    //for(int i = 0; i < 1; i++) state.act_H(i);  
    // std::string str = "HTSHTSHTSHTHTHTHTSHTHTSHTSHTSHTHTHTSHTSHTHTHTSHTHTSHTHTHTHTHTHTHTSHTSHTSHTHTSHTHTSHTHTHTHTSHTHTHTSHTHTSHTHTHTHTSHTSHTSHTHTHTSHTSHTSHTSHTHTSHTSHTSHTSHTHTSHTHTSHTSHTHTHTHTHTSHTHTHTHTSHTSHTSHTHTSHTSHTHTHTSHTHTHTHTHTSHTSHTHTHTHTHTSHTHTHTHTSHTHTHTHTHTHTH";
    // for(int i = 0; i < str.size(); i++){
    //     if(str[i] == 'H') state.act_H(0);
    //     else{
    //         state.act_T(0);
    //         if(str[i] == 'S') state.act_T(0);
    //     }
    // }
    //std::cout << state << std::endl;
}