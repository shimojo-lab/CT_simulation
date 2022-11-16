#include <iostream>
#include <fstream>
#include "state.cu"
#include "exist_state.cu"

int main(){
    int n = 30;
    cudaSetDevice(1);
    QuantumState state(n);
    exist_QuantumState exist_state(n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time, exist_time;


    std::ofstream H_ofs("H_gate.csv");
    for(int i = 0; i < n; i++){
        // 提案手法の時間計測
        cudaEventRecord(start);
        state.act_H(i);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        // 既存手法の時間計測
        cudaEventRecord(start);
        exist_state.act_H(i);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&exist_time, start, stop);

        H_ofs << "H" << i << ", " << time << ", " << exist_time << ", "  << time / exist_time << ", "<< std::endl;
    }

    std::ofstream S_ofs("S_gate.csv");
    for(int i = 0; i < n; i++){
        // 提案手法の時間計測
        cudaEventRecord(start);
        state.act_S(i);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        // 既存手法の時間計測
        cudaEventRecord(start);
        exist_state.act_S(i);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&exist_time, start, stop);

        S_ofs << "H" << i << ", " << time << ", " << exist_time << ", "  << time / exist_time << ", "<< std::endl;
    }

    std::ofstream T_ofs("T_gate.csv");
    for(int i = 0; i < n; i++){
        // 提案手法の時間計測
        cudaEventRecord(start);
        state.act_T(i);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        // 既存手法の時間計測
        cudaEventRecord(start);
        exist_state.act_T(i);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&exist_time, start, stop);

        T_ofs << "H" << i << ", " << time << ", " << exist_time << ", "  << time / exist_time << ", "<< std::endl;
    }

    cudaEventRecord(start);
    state.act_CNOT(0, 1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "CNOT " << time << std::endl;

    CTYPE* vec = state.get_vec();
    std::cout << vec[0] << std::endl;
    return 0;
}
