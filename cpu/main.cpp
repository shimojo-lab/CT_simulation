#include <complex>
#include <string>
#include <math.h>
#include <iostream>
#include "state.hpp"

int main()
{
    int n = 30;
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
