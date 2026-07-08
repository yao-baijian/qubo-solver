#include <Eigen/Dense>
// #include <ATen/SparseTensorUtils.h>
#include <torch/torch.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include "BsbSolver.hh"
#include "AlgebraicSolver.hh"

#include "Sparse.hh"

namespace Sat {

class SatSolver 
{

public: 
    SatSolver(SatOptions *satOption, Sb::BsbSolver * bsbSolver, AlgebraicSolver * algebraicSolver)
    : _options(satOption), _currCnfPtr(nullptr), _bsbSolver(bsbSolver), _algebraicSolver(algebraicSolver) {};

    ~SatSolver() = default;

    void solve(const std::string& filename);

private:

    // read cnf file
    void readFile(const std::string& filename);

    void addAuxBitDim1(int var1, bool sign);

    void addAuxBitDim3(int var1, int var2, int var3, bool sign);

    // TODO MAX-k-SAT to 3-SAT
    std::vector<Clause> kSat23Sat(Clause &clause);

    Eigen::MatrixXd sat2Qubo();

    // convert 3-SAT to pubo
    std::pair<Eigen::MatrixXd,  Eigen::Tensor<int, 3>> sat2Pubo();

    std::pair<torch::Tensor, torch::Tensor> sat2PuboSparse();

    void reportSol(Eigen::VectorXd &sol);

    void legalize(Eigen::VectorXd &sol);

    Eigen::MatrixXd pyCubicOptimizerWrapper(const Eigen::Tensor<int, 3> &A, const Eigen::MatrixXd &B, const int dim);
    // Eigen → PyTorch
    torch::Tensor eigen_to_torch(const Eigen::Tensor<int, 3>& e_tensor);

    torch::Tensor eigen_to_torch(const Eigen::MatrixXd& eigen_mat);
    // PyTorch → Eigen
    Eigen::Tensor<int, 3> torch_to_eigen(const torch::Tensor& t_tensor);

    void clean();

    CNF _Cnfs, _kto3Cnfs, _QbitCnfs, _penaltyCnfs;

    CNF *_currCnfPtr;

    Sb::BsbSolver *_bsbSolver;

    SatOptions *_options;

    AlgebraicSolver *_algebraicSolver;

};

}