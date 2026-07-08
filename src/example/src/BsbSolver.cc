#include <fstream>
#include <random>
#include <iostream>

#include "define.hh"
#include "BsbSolver.hh"
#include "matplotlibcpp.h" 

namespace plt = matplotlibcpp;

namespace Sb {

//  GPU wrap
#ifdef CUDA_ENABLE
#include <GEAM.h>

extern "C" Eigen::VectorXd bsbCuda(
    const Eigen::MatrixXd& H, 
    const bool dynaHessian, 
    const Eigen::Tensor<int, 3> &A, 
    const int num_iters, 
    const double dt
);

extern "C" Eigen::VectorXd bsbCudaSparse(
    const torch::Tensor& H_sparse,
    const bool dynaHessian,
    const torch::Tensor& A_sparse,     
    const int num_iters,
    const double dt
);

#endif


// CPU 
void
BsbSolver::genH(Eigen::MatrixXd &H, const std::vector<Eigen::MatrixXd> &cost)
{   
    const auto costNum = cost.size();

    double coeff[costNum] = {1.0, 2.0, 2.0, 3.0};
    std::string name[costNum] = { "hpwl diff", "area diff", "width diff", "height diff"};

    for (int i = 0; i < costNum; ++i) { 
        H += coeff[i] * cost[i] / (cost[i].maxCoeff() - cost[i].minCoeff());
        logWeight(name[i], cost[i]);
    }
    H = -H;
    Eigen::MatrixXd transH = H.transpose();

    std::cout << "\nweighted edge [10 x 10]:\n" << H.topLeftCorner(10, 10) << std::endl;

    H = H + transH;
}

void
BsbSolver::genH(Eigen::MatrixXd &H, const std::vector<Eigen::MatrixXd> &cost, 
    const std::vector<Eigen::MatrixXd> &penalty)
{   
    const auto costNum = cost.size();
    double coeff[costNum] = {1.0, 2.0, 2.0, 3.0};

    for (int i = 0; i < costNum; ++i) { 
        H += coeff[i] * cost[i] / (cost[i].maxCoeff() - cost[i].minCoeff());
    }

    const auto penaltyNum = cost.size();

    for (int i = 0; i < penaltyNum; ++i) { 
        H += penalty[i];
    }

    H = -H;
    Eigen::MatrixXd transH = H.transpose();

    std::cout << H.topLeftCorner(15, 15) << std::endl;

    H = H + transH;
}

void
BsbSolver::logWeight(std::string name, Eigen::MatrixXd W) 
{
    std::cout <<"Max " << name << ": " << W.maxCoeff() << " | Min " << name << ": " << W.minCoeff() << std::endl;
}

Eigen::VectorXd
BsbSolver::solve(const Eigen::MatrixXd &H, const bool dynaHessian, const Eigen::Tensor<int, 3> &A) 
{
#ifdef CUDA_ENABLE
    if (_options->solverMode == SolverMode::GPU) {
        return bsbCuda(H, dynaHessian, A, _options->iterNum, _options->dt);
    }
#endif
    return solveCpu(H);
}

Eigen::VectorXd
BsbSolver::solve(const Eigen::SparseMatrix <double> &H, const torch::Tensor &tensorH, const bool dynaHessian, const torch::Tensor &tensorA) 
{
#ifdef CUDA_ENABLE
    if (_options->solverMode == SolverMode::GPU) { // pending for GPU sparse
        return bsbCudaSparse(tensorH, dynaHessian, tensorA, _options->iterNum, _options->dt);
    } 
#endif
    return solveCpu(H);
}

Eigen::VectorXd
BsbSolver::solveCpu(const Eigen::MatrixXd &H)
{
    auto dim = H.rows();

    Eigen::VectorXd x_comp = 0.1 *Eigen::VectorXd::Random(dim);
    Eigen::VectorXd y_comp = 0.1 *Eigen::VectorXd::Random(dim); 

    double xi = 0.7 / std::sqrt(dim); 

    Eigen::VectorXd sol = x_comp.array().sign(); 
    std::vector<double> energies;
    double e = -0.5 * sol.transpose() * H * sol;
    auto energy = -0.25 * H.sum() - 0.5 * e;
    energies.emplace_back(energy);

    std::vector<double> alpha(_options->iterNum);
    for (int i = 0; i < _options->iterNum; ++i) {
        alpha[i] = static_cast<double>(i) / (_options->iterNum - 1);
    }
    
    for (int i = 0; i < _options->iterNum; ++i) {
        y_comp += ((-1 + alpha[i]) * x_comp + xi * (H * x_comp)) * _options->dt;
        x_comp += y_comp * _options->dt;
        for (int j = 0; j < x_comp.size(); ++j) {
            if (std::abs(x_comp[j]) > 1) {
                y_comp[j] = 0.0;
            }
        }
        x_comp = x_comp.cwiseMax(-1).cwiseMin(1);

        sol = x_comp.array().sign();
        e = - 0.5 * sol.transpose() * H * sol;
        energy = -0.25 * H.sum() - 0.5 * e;
        energies.emplace_back(energy);
    }
    plot(energies);
    return sol;
}

Eigen::VectorXd
BsbSolver::solveCpu(const Eigen::SparseMatrix <double> &H)
{
    auto dim = H.rows();

    Eigen::VectorXd x_comp = 0.1 *Eigen::VectorXd::Random(dim);
    Eigen::VectorXd y_comp = 0.1 *Eigen::VectorXd::Random(dim); 

    double xi = 0.7 / std::sqrt(dim); 

    Eigen::VectorXd sol = x_comp.array().sign(); 
    std::vector<double> energies;
    double e = -0.5 * sol.transpose() * H * sol;
    auto energy = -0.25 * H.sum() - 0.5 * e;
    energies.emplace_back(energy);

    std::vector<double> alpha(_options->iterNum);
    for (int i = 0; i < _options->iterNum; ++i) {
        alpha[i] = static_cast<double>(i) / (_options->iterNum - 1);
    }
    
    for (int i = 0; i < _options->iterNum; ++i) {
        y_comp += ((-1 + alpha[i]) * x_comp + xi * (H * x_comp)) * _options->dt;
        x_comp += y_comp * _options->dt;
        for (int j = 0; j < x_comp.size(); ++j) {
            if (std::abs(x_comp[j]) > 1) {
                y_comp[j] = 0.0;
            }
        }
        x_comp = x_comp.cwiseMax(-1).cwiseMin(1);

        sol = x_comp.array().sign();
        e = - 0.5 * sol.transpose() * H * sol;
        energy = -0.25 * H.sum() - 0.5 * e;
        energies.emplace_back(energy);
    }
    plot(energies);
    return sol;
}

Eigen::MatrixXd 
BsbSolver::legalize(const Eigen::MatrixXd &weight, const Eigen::VectorXd &sol, const size_t dim) {
    Eigen::MatrixXd solMap = Eigen::MatrixXd::Zero(dim, dim);

    for (int i = 0; i < dim; ++i) {
        int currSpinStat = (sol(i*dim) > 0) ? 1 : -1;

        auto swapNum = 0;
        for (int j = i; j < dim; ++j) {
            
            int swapSpinStat = (sol(i*dim + j) > 0) ? 1 : -1;
            int swap = 1 - currSpinStat * swapSpinStat;

            if (swap == 2) {
                swapNum ++; 
                solMap (i, j) = 1;
            } else if (swap == 0) {
                continue;
            } else {
                std::cerr << "solution out of bound" << std::endl;
                exit(0);
            }
        }
    }

    // analysis exchanged result

    for (int i = 0; i < dim; ++i) {
        bool foundFirstOne = false;
        for (int j = i; j < dim; ++j) {
            if (solMap(i, j) == 1.0) {
                if (weight(i,j) < 0 && !foundFirstOne) {
                    foundFirstOne = true;
                } else {
                    solMap(i, j) = 0.0;
                }
            }
        }
    }

    std::cout << "solution:\n" << solMap << std::endl;

    return solMap;
}

void
BsbSolver::plot(std::vector<double> energies)
{
    plt::xlabel("iterations");
    plt::ylabel("Ising Energy");

    plt::plot(energies);
    plt::show();
}

void
BsbSolver::scaleMatrix(Eigen::MatrixXd &H, size_t size) 
{   
    for (int i = 0; i < H.rows(); ++i) {
        for (int j = 0; j < H.cols(); ++j) {
            if (H(i, j) != 0) {
                H(i, j) = H(i, j) * (rand() % size);
            }
        }
    }
}

Eigen::MatrixXd
BsbSolver::loadData(const std::string &name) 
{
    std::ifstream file(name);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << name << std::endl;
        exit(1);
    }

    std::string line;
    int N = 0;
    Eigen::MatrixXd J;

    for (int idx = 0; std::getline(file, line); ++idx) {
        std::istringstream iss(line);
        if (idx == 0) {
            iss >> N;
            J = Eigen::MatrixXd::Zero(N, N);
        } else {
            int i, j;
            double value;
            iss >> i >> j >> value;
            J(i - 1, j - 1) = value;
        }
    }

    file.close();
    Eigen::MatrixXd H = -J;

    std::cout << (H.array() != 0).count() << std::endl;

    return H;
}

}