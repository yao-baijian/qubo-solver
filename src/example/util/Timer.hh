    
#include <iostream>
#include <chrono>
#include <vector>
    
// CPU timer
struct CpuTimer {
    using Clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<Clock> start;
    std::chrono::time_point<Clock> end;
    void recordStart() { start = Clock::now(); }
    float recordStop() {
        end = Clock::now();
        return std::chrono::duration<float, std::milli>(end - start).count();
    }
    void report() {
        std::cout << "[Timer] CPU time: " << std::chrono::duration<float, std::milli>(end - start).count() << " ms\n" << std::endl;
    }
};

    
// void run_computation() {
//     CpuTimer cpu_timer;
//     cpu_timer.recordStart();
//     for (int i = 0; i < 1000000; ++i) { }
//     float cpu_time = cpu_timer.recordStop();

//     std::cout << "CPU Time: " << cpu_time << " ms\n"
//               << "Total Time: " << cpu_time << " ms" << std::endl;
// }


// auto *algebraicSolver = new AlgebraicSolver();
// auto result = algebraicSolver->solve({15, 4});
// std::cout << "b15, b4: [";
// for (auto coeff : result) {
//     std::cout << coeff << " ";
// }
// std::cout << "]" << std::endl;

// result = algebraicSolver->solve({-125, -2});
// std::cout << "~b125, ~b2: [";
// for (auto coeff : result) {
//     std::cout << coeff << " ";
// }
// std::cout << "]" << std::endl;

// result = algebraicSolver->solve({-1, 2, 3});
// std::cout << "~b1, b2, b3: [";
// for (auto coeff : result) {
//     std::cout << coeff << " ";
// }
// std::cout << "]" << std::endl;  

// result = algebraicSolver->solve({-44, -2, 69});
// std::cout << "~b44, ~b2, b69: [";
// for (auto coeff : result) {
//     std::cout << coeff << " ";
// }
// std::cout << "]" << std::endl;

// result = algebraicSolver->solve({-5, -7, -11});
// std::cout << "~b5, ~b7, ~b11: [";
// for (auto coeff : result) {
//     std::cout << coeff << " ";
// }
// std::cout << "]" << std::endl;  

// exit(0);