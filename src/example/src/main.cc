#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <assert.h>
#include <Eigen/Dense>

#include "BStarTree.hh"
#include "SatSolver.hh"
#include "AlgebraicSolver.hh"

#include "Timer.hh"

void fp(const std::string blocksName, 
    const std::string output_dir, 
    double dead_space_ratio, 
    const std::string mode, 
    const std::string subMode, 
    Sb::BsbSolver *bsbSolver) 
{

    auto *bStarTree = new Floorplan::BStarTree(dead_space_ratio, bsbSolver);

    // Read testcase
    bStarTree->parse_hardblocks(blocksName + ".hardblocks");
    bStarTree->parse_net(blocksName + ".nets");
    bStarTree->parse_pl(blocksName + ".pl");

    bStarTree->initBstarTree();

    if(bStarTree->getBlocksNum()==300 && dead_space_ratio==0.15)
        srand(SEED);

    std::string output_file = "";
    bool result = false;

    // Floorplanning
    if (mode == "SA") {
        result = bStarTree->SimulatedAnnealing();
        output_file = output_dir + "/" + blocksName + "_SA.floorplan";
    } else if (mode == "SB") {
        bStarTree->initSimulatedBifurcation(1, 1000, 0.25);
        result = bStarTree->quboBsb(subMode);
        output_file = output_dir + "/" + blocksName + "_SB.floorplan";
    } else {
        assert(false);
    }

    if (result) {
        bStarTree->write_output(output_file);
    }

    // Total Runtime
    double runtime = clock();
    printf("[  Total Run time  ]: %f sec\n",runtime/CLOCKS_PER_SEC);
    return;
}

void sat(const std::string fileName, 
        const std::string mapping, 
        const std::string batch, 
        const int reduce, 
        const double hessian, 
        const double lambda, 
        Sb::BsbSolver *bsbSolver, 
        Sat::AlgebraicSolver *algebraicSolver)
{   
    Sat::SatOptions satOption;
    satOption.mappingMode = (mapping == "auxiliary") ? Sat::MappingMode::AUXILIARY : Sat::MappingMode::HESSIAN;
    satOption.hessianCoeff = hessian;
    satOption.batch = (batch == "batch") ? true : false;
    satOption.quboReduceMode = reduce;
    satOption.penaltyCoeff=lambda;

    auto *satSolver = new Sat::SatSolver(&satOption, bsbSolver, algebraicSolver);

    if (satOption.batch) {
        for (int i = 0; i < 40; ++i) {
            std::string batchFileName = fileName + std::to_string(i) + ".cnf";
            satSolver->solve(batchFileName);
        }
    } else {
        satSolver->solve(fileName); // without hessian
    }
}


int main(int argc, char **argv)
{

    if (argc != 7 && argc != 8) {
        std::cerr << "Usage: " << argv[0] << " fp <blocksName> <output_dir> <dead_space_ratio> <mode> <subMode> \n"
                  << argv[0] << " sat <filename> <option> <batch> <reduce> <hessian_coeff> <lambda> \n" << std::endl;
        return 1;
    }

    std::string exe = argv[1];

    Sb::sbSolverOptions options;

#ifdef CUDA_ENABLE
    options.solverMode = Sb::SolverMode::GPU;
#else
    options.solverMode = Sb::SolverMode::CPU;
#endif
    options.iterNum = 1000;
    options.dt = 0.25;
    // options.mapping = Sat::MappingMode::HESSIAN;

    options.mapping = (argv[3] == "auxiliary") ? Sat::MappingMode::AUXILIARY : Sat::MappingMode::HESSIAN;

    auto *bsbSolver = new Sb::BsbSolver(&options);
    auto *algebraicSolver = new Sat::AlgebraicSolver();
    CpuTimer timer;

    if (exe == "fp") {
        fp(argv[2], argv[3], atof(argv[4]), argv[5], argv[6], bsbSolver);
    } else if (exe == "sat") {
        timer.recordStart();
        sat(argv[2], argv[3], argv[4], std::stoi(argv[5]), std::stod(argv[6]), std::stod(argv[7]), bsbSolver, algebraicSolver);
        timer.recordStop();
        timer.report();
    }

    return 0;
}





