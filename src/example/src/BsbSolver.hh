#ifndef BSBSOLVER_HH
#define BSBSOLVER_HH

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>
#include <torch/torch.h>

#include "define.hh"

namespace Sb {

class BsbSolver
{

public: 
    BsbSolver(sbSolverOptions *options) 
    : _options(options) {};

    ~BsbSolver() = default;

    // Eigen::VectorXd solve(const Eigen::MatrixXd &H);
    // Eigen::VectorXd solve(const Eigen::SparseMatrix <double> &H);
    
    Eigen::VectorXd solve(const Eigen::MatrixXd &H, const bool dynaHessian, const Eigen::Tensor<int, 3> &A);
    Eigen::VectorXd solve(const Eigen::SparseMatrix <double> &H, const torch::Tensor &tensorH, const bool dynaHessian, const torch::Tensor &tensorA);
    void genH(Eigen::MatrixXd &H, const std::vector<Eigen::MatrixXd> &cost);
    void genH(Eigen::MatrixXd &H, const std::vector<Eigen::MatrixXd> &cost, 
        const std::vector<Eigen::MatrixXd> &penalty);
    Eigen::MatrixXd legalize(const Eigen::MatrixXd &weight, const Eigen::VectorXd &sol, const size_t dim);

private:

    // dense matrix
    Eigen::VectorXd solveCpu(const Eigen::MatrixXd &H);
    // cpu sparse matrix accel
    Eigen::VectorXd solveCpu(const Eigen::SparseMatrix <double> &H);
    Eigen::MatrixXd loadData(const std::string& name);

    void plot(std::vector<double> energies);
    void scaleMatrix(Eigen::MatrixXd &H, size_t size);
    void logWeight(std::string name, Eigen::MatrixXd W);
    sbSolverOptions *_options;

};

} // namespace Sb

#endif