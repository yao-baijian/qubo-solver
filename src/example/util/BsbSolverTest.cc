#include "define.hh"
#include "BsbSolver.hh"

int main() {
    Sb::sbSolverOptions options;

#ifdef CUDA_ENABLE
    options.solverMode = Sb::SolverMode::GPU;
#else
    options.solverMode = Sb::SolverMode::CPU;
#endif
    options.iterNum = 1000;
    options.dt = 0.25;
    options.mapping = Sat::MappingMode::HASSIAN;

    auto *bsbSolver = new Sb::BsbSolver(&options);

    auto name = "/home/byao/Desktop/qsbm/data/Gset/G12";
    auto H = loadData(name);
    scaleMatrix(H, 10);
    auto hNew = H.topLeftCorner(200, 200);
    // solve(H);

    // solve(hNew);
}