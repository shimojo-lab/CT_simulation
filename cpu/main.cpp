#include <iostream>
#include <fstream>
#include <time.h>
#include "state.hpp"
#include "exist_state.hpp"

int main(){
    int n = 25;
    QuantumState state(n);
    exist_QuantumState exist_state(n);

    clock_t start, end;
    double time, exist_time;


    std::ofstream H_ofs("H_gate.csv");
    for(int i = 0; i < n; i++){
        // 提案手法の時間計測
        start = clock();
        state.act_H(i);
        end = clock();
        time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;

        // 既存手法の時間計測
        start = clock();
        exist_state.act_H(i);
        end = clock();
        exist_time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;

        H_ofs << "H" << i << ", " << time << ", " << exist_time << ", "  << time / exist_time << ", "<< std::endl;
    }

    std::ofstream S_ofs("S_gate.csv");
    for(int i = 0; i < n; i++){
        // 提案手法の時間計測
        start = clock();
        state.act_S(i);
        end = clock();
        time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;

        // 既存手法の時間計測
        start = clock();
        exist_state.act_S(i);
        end = clock();
        exist_time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;

        S_ofs << "S" << i << ", " << time << ", " << exist_time << ", "  << time / exist_time << ", "<< std::endl;
    }

    std::ofstream T_ofs("T_gate.csv");
    for(int i = 0; i < n; i++){
        // 提案手法の時間計測
        start = clock();
        state.act_T(i);
        end = clock();
        time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;

        // 既存手法の時間計測
        start = clock();
        exist_state.act_T(i);
        end = clock();
        exist_time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;

        T_ofs << "T" << i << ", " << time << ", " << exist_time << ", "  << time / exist_time << ", "<< std::endl;
    }

    CTYPE* vec = state.get_vec();
    exist_CTYPE* vec_exist = exist_state.get_vec();
    // std::cout << vec[0] << std::endl;
    return 0;
}
