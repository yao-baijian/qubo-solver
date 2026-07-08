#include <fstream>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/eigen.h>  // 用于Tensor转换

#include "SatSolver.hh"

namespace py = pybind11;

namespace Sat {

void
SatSolver::solve(const std::string& filename) 
{
    clean();
    readFile(filename);
    std::cout << "[Report] read file" << filename << std::endl;
    std::cout << "[Report] var num: " << _Cnfs.varNum << ", clause num: " << _Cnfs.clauseNum << std::endl;

    if (_options->mappingMode == MappingMode::AUXILIARY) {
        // Convert k-SAT to 3-SAT
        _kto3Cnfs.varNum = _Cnfs.varNum;
        for (auto &clause : _Cnfs.clauses) {
            if (clause.size() > 3) {
                auto generatedClauses = kSat23Sat(clause);
                _kto3Cnfs.clauses.insert(_kto3Cnfs.clauses.end(), generatedClauses.begin(), generatedClauses.end());
            } else {
                _kto3Cnfs.clauses.emplace_back(clause);
            }
        }
        _kto3Cnfs.clauseNum = _kto3Cnfs.clauses.size();
        std::cout << "[Report] kto3 var num: " << _kto3Cnfs.varNum << ", clause num: " << _kto3Cnfs.clauseNum << std::endl;

        // Convert 3-SAT to QUBO
        _QbitCnfs.varNum = _kto3Cnfs.varNum;
        _currCnfPtr = &_kto3Cnfs;
        auto H = sat2Qubo();
        _QbitCnfs.clauseNum = _QbitCnfs.clauses.size();
        std::cout << "[Report] QUBO var num: " << _QbitCnfs.varNum << ", clause num: " << _QbitCnfs.clauseNum << std::endl;

        Eigen::Tensor<int, 3> A;
        auto sol = _bsbSolver->solve(H, false, A);

        reportSol(sol);
        legalize(sol);

    } else if (_options->mappingMode == MappingMode::HESSIAN) {
        // Convert k-SAT to 3-SAT
        _kto3Cnfs.varNum = _Cnfs.varNum;
        for (auto &clause : _Cnfs.clauses) {
            if (clause.size() > 3) {
                auto generatedClauses = kSat23Sat(clause);
                _kto3Cnfs.clauses.insert(_kto3Cnfs.clauses.end(), generatedClauses.begin(), generatedClauses.end());
            } else {
                _kto3Cnfs.clauses.emplace_back(clause);
            }
        }
        _kto3Cnfs.clauseNum = _kto3Cnfs.clauses.size();
        std::cout << "[Report] kto3 var num: " << _kto3Cnfs.varNum << ", clause num: " << _kto3Cnfs.clauseNum << std::endl;

        // Convert 3-SAT to PUBO
        _QbitCnfs.varNum = _kto3Cnfs.varNum;
        _currCnfPtr = &_kto3Cnfs;
        auto hA = sat2Pubo();
        _QbitCnfs.clauseNum = _QbitCnfs.clauses.size();
        std::cout << "[Report] QUBO var num: " << _QbitCnfs.varNum << ", clause num: " << _QbitCnfs.clauseNum << std::endl;
        std::cout << "[Report] H dimension" << hA.first.rows() << std::endl;
        std::cout << "[Report] A dimension" << hA.second.dimension(0) << std::endl;
        auto sol = _bsbSolver->solve(hA.first, true, hA.second);

        reportSol(sol);
        legalize(sol);
    } else {
        std::cerr << " Mode unspecified" << std::endl;
    }   

}

void
SatSolver::readFile(const std::string& filename)
{   
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == 'c') {
            continue; 
        }
        if (line[0] == 'p') {
            std::istringstream iss(line);
            std::string token;
            iss >> token; 
            iss >> token; 
            if (token != "cnf") {
                std::cerr << "PARSE ERROR! Expected 'p cnf', got: " << line << std::endl;
                return;
            }
            iss >> _Cnfs.varNum >> _Cnfs.clauseNum;
            break;
        }
    }

    if (_Cnfs.varNum == 0 || _Cnfs.clauseNum == 0) {
        std::cerr << "PARSE ERROR! No valid 'p cnf' line found." << std::endl;
        return;
    }

    for (int i = 0; i < _Cnfs.clauseNum; i++) {

        std::getline(file, line);
        std::istringstream iss(line);
        std::vector<int> clause;
        int literal;

        while (iss >> literal && literal != 0) {
            clause.emplace_back(literal);
        }

        _Cnfs.clauses.emplace_back(clause);
    }

    file.close();
}

// qubo conversion
Eigen::MatrixXd
SatSolver::sat2Qubo()
{

    Eigen::VectorXd V = Eigen::VectorXd::Zero(_currCnfPtr->varNum);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(_currCnfPtr->varNum, _currCnfPtr->varNum);

    std::map<std::pair<uint32_t, uint32_t>, int> penaltyMap;

    for (const auto& clause : _currCnfPtr->clauses) {
        // encoding using choi's representation
        // xi => xi   ~xi => 1 - xi

        auto coeffTerms = _algebraicSolver->solve(clause);

        for (const auto& [term, coeff] : coeffTerms) {
            bool sign = (coeff > 0) ? true : false;

            if (term.size() == 1) {
                addAuxBitDim1(term[0], sign);

            } else if (term.size() == 2) {
                H(term[0] - 1, term[1] - 1) = sign ? 1.0 : -1.0;
            } else if (term.size() == 3) {        
                /*1 a(xi + xj + xk - 2);
                * 2 w = x1x2x3
                *   H penalty = λ(w(x1​+x2​+x3​−2)−x1​x2​−x1​x3​−x2​x3​+3w)
                * 3 x1​x2​x3 ​= x1​x2​+x1​x3​+x2​x3​−2(x1​+x2​+x3​)+3w
                *   H penalty = 
                * 4 x1​x2​x3 ​= (x1​x2​)x3​; w12 = x1x2
                *   H penalty ​= λ(x1​x2​−w12​(x1​+x2​−1)).
                */

                if (_options->quboReduceMode == 1) {
                    addAuxBitDim3(term[0], term[1], term[2], sign);
                } else if (_options->quboReduceMode == 2) {  
                    // penalty : wn*x1; wn*x2; wn*x3; wn => wn * wn+1; -x1x2; -x1x3; -x2x3; 
                    
                    int aux1 = _QbitCnfs.varNum + 1;
                    int aux2 = _QbitCnfs.varNum + 2;
                    _QbitCnfs.varNum += 2;

                    _QbitCnfs.clauses.emplace_back(Clause{aux1, aux2});

                    penaltyMap.emplace(std::make_pair(aux1, term[0]), 1);
                    penaltyMap.emplace(std::make_pair(aux1, term[1]), 1);
                    penaltyMap.emplace(std::make_pair(aux1, term[2]), 1);

                    penaltyMap.emplace(std::make_pair(term[0], term[1]), -1);
                    penaltyMap.emplace(std::make_pair(term[0], term[2]), -1);
                    penaltyMap.emplace(std::make_pair(term[1], term[2]), -1);
                    
                    penaltyMap.emplace(std::make_pair(aux1, aux2), 1);

                } else if (_options->quboReduceMode == 3) { 

                    _QbitCnfs.varNum ++;
                    // this might cancel with previous assigment
                    // H(term[0] - 1, term[1] - 1) = sign ? 1.0 : -1.0;
                    // H(term[1] - 1, term[2] - 1) = sign ? 1.0 : -1.0;
                    // H(term[0] - 1, term[2] - 1) = sign ? 1.0 : -1.0;

                    // add ampititude 
                    // addAuxBitDim1(term[0], -sign);
                    // addAuxBitDim1(term[1], -sign);
                    // addAuxBitDim1(term[2], -sign);

                    // addAuxBitDim1(_QbitCnfs.varNum, sign);

                } else if (_options->quboReduceMode == 4) {
                    // penalty : -wn*x1; -wn*x2; wn; x1x2;
                    int aux1 = _QbitCnfs.varNum + 1;
                    int aux2 = _QbitCnfs.varNum + 2;

                    _QbitCnfs.varNum += 2;
                    _QbitCnfs.clauses.emplace_back(Clause{aux1, term[2]});

                    penaltyMap.emplace(std::make_pair(aux1, term[0]), -1);
                    penaltyMap.emplace(std::make_pair(aux1, term[1]), -1);

                    penaltyMap.emplace(std::make_pair(term[0], term[1]), 1);
                    penaltyMap.emplace(std::make_pair(aux1, aux2), 1);

                }

            }
        }
    }

    // std::cout << H << std::endl;

    Eigen::MatrixXd newH = Eigen::MatrixXd::Zero(_QbitCnfs.varNum, _QbitCnfs.varNum);
    newH.topLeftCorner(_currCnfPtr->varNum, _currCnfPtr->varNum) = H;

    if (_QbitCnfs.varNum > _currCnfPtr->varNum) {
        for (const auto& clause : _QbitCnfs.clauses) { 
            newH(abs(clause[0]) - 1, abs(clause[1]) - 1) = (clause[0] > 0) ? 1.0: -1.0;
        }
    }

    if (_options->quboReduceMode >= 2) {
        // possible penalty term
        Eigen::MatrixXd penaltyH = Eigen::MatrixXd::Zero(_QbitCnfs.varNum, _QbitCnfs.varNum);
        for (const auto& [key, val] : penaltyMap) { 
            penaltyH(key.first - 1, key.second - 1) = val;
        }
        newH = newH + _options->penaltyCoeff * penaltyH;
    }

    return - newH - newH.transpose();
}

std::pair<Eigen::MatrixXd,  Eigen::Tensor<int, 3>>
SatSolver::sat2Pubo()
{

    Eigen::VectorXd V = Eigen::VectorXd::Zero(_currCnfPtr->varNum);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(_currCnfPtr->varNum, _currCnfPtr->varNum);
    Eigen::Tensor<int, 3> A (_currCnfPtr->varNum, _currCnfPtr->varNum, _currCnfPtr->varNum);
    A.setZero();

    for (const auto& clause : _currCnfPtr->clauses) {
        auto coeffTerms = _algebraicSolver->solve(clause);

        for (const auto& [term, coeff] : coeffTerms) {
            bool sign = (coeff > 0) ? true : false;

            if (term.size() == 1) {
                addAuxBitDim1(term[0], sign);
            } else if (term.size() == 2) {
                H(term[0] - 1, term[1] - 1) = sign ? 1 : -1;
            } else if (term.size() == 3) {        
                A(term[0] - 1, term[1] - 1, term[2] - 1) = sign ? 1 : -1;
                A(term[0] - 1, term[2] - 1, term[1] - 1) = sign ? 1 : -1;
                A(term[1] - 1, term[2] - 1, term[0] - 1) = sign ? 1 : -1;

                A(term[1] - 1, term[0] - 1, term[2] - 1) = sign ? 1 : -1;
                A(term[2] - 1, term[0] - 1, term[1] - 1) = sign ? 1 : -1;
                A(term[2] - 1, term[1] - 1, term[0] - 1) = sign ? 1 : -1;
            }
        }
    }

    Eigen::MatrixXd newH = Eigen::MatrixXd::Zero(_QbitCnfs.varNum, _QbitCnfs.varNum);
    newH.topLeftCorner(_currCnfPtr->varNum, _currCnfPtr->varNum) = H;

    if (_QbitCnfs.varNum > _currCnfPtr->varNum) {
        for (const auto& clause : _QbitCnfs.clauses) { 
            newH(abs(clause[0]) - 1, abs(clause[1]) - 1) = (clause[0] > 0) ? 1.0: -1.0;
        }
    }

    return std::make_pair(- newH - newH.transpose(), -A);
}

// Sparse support using libtorch
std::pair<torch::Tensor, torch::Tensor>
SatSolver::sat2PuboSparse() {
    int varNum = _currCnfPtr->varNum;
    
    // H init
    std::vector<int64_t> H_indices;  // coordinate list (2 x nnz)
    std::vector<double> H_values;    // value list
    H_indices.reserve(2 * varNum * varNum);
    H_values.reserve(varNum * varNum);

    // tensor A init
    std::vector<int64_t> A_indices;  // (3 x nnz)
    std::vector<int64_t> A_values;   // 
    A_indices.reserve(3 * varNum * varNum * varNum);
    A_values.reserve(varNum * varNum * varNum);

    for (const auto& clause : _currCnfPtr->clauses) {
        auto coeffTerms = _algebraicSolver->solve(clause);

        for (const auto& [term, coeff] : coeffTerms) {
            bool sign = (coeff > 0);
            int value = sign ? 1 : -1;

            if (term.size() == 1) {
                addAuxBitDim1(term[0], sign);
            } else if (term.size() == 2) {
                int i = term[0] - 1, j = term[1] - 1;
                H_indices.push_back(i); H_indices.push_back(j);
                H_values.push_back(value);
            } else if (term.size() == 3) {
                int i = term[0] - 1, j = term[1] - 1, k = term[2] - 1;
                A_indices.insert(A_indices.end(), {i, j, k}); A_values.push_back(value);
                A_indices.insert(A_indices.end(), {i, k, j}); A_values.push_back(value);
                A_indices.insert(A_indices.end(), {j, i, k}); A_values.push_back(value);
                A_indices.insert(A_indices.end(), {j, k, i}); A_values.push_back(value);
                A_indices.insert(A_indices.end(), {k, i, j}); A_values.push_back(value);
                A_indices.insert(A_indices.end(), {k, j, i}); A_values.push_back(value);
            }
        }
    }

    // 3. use libtorch coo tensor
    auto H_indices_tensor = torch::tensor(H_indices, torch::kInt64).reshape({2, static_cast<int64_t>(H_values.size())});
    auto H_values_tensor = torch::tensor(H_values, torch::kFloat64);
    auto H = torch::sparse_coo_tensor(
        H_indices_tensor,
        H_values_tensor,
        {varNum, varNum},
        torch::kDouble
    ).coalesce(); // combine replicate term

    // 4. process auxiliary bit
    if (_QbitCnfs.varNum > varNum) {
        for (const auto& clause : _QbitCnfs.clauses) {
            int i = abs(clause[0]) - 1, j = abs(clause[1]) - 1;
            double value = (clause[0] > 0) ? 1.0 : -1.0;
            H_indices_tensor = torch::cat({H_indices_tensor, 
                torch::tensor({i, j}, torch::kInt64).reshape({2, 1})}, 1);
            H_values_tensor = torch::cat({H_values_tensor, 
                torch::tensor({value}, torch::kFloat64)});
        }
        H = torch::sparse_coo_tensor(
            H_indices_tensor,
            H_values_tensor,
            {_QbitCnfs.varNum, _QbitCnfs.varNum},
            torch::kDouble
        ).coalesce();
    }

    // 5. negative
    auto H_T = H.transpose(0, 1).coalesce();
    auto negSymH = H.add(H_T).mul(-1).coalesce();

    // 6. libtorch tensor A
    auto A_indices_tensor = torch::tensor(A_indices, torch::kInt64).reshape({3, static_cast<int64_t>(A_values.size())});
    auto A_values_tensor = torch::tensor(A_values, torch::kInt64);
    auto A = torch::sparse_coo_tensor(
        A_indices_tensor,
        A_values_tensor,
        {varNum, varNum, varNum},
        torch::kInt64
    ).coalesce();
    auto negA = A.mul(-1).coalesce();

    return std::make_pair(negSymH, negA);
}

// sat textbook conversion
std::vector<Clause>
SatSolver::kSat23Sat(Clause &clause)
{   
    std::vector<Clause> tempCnfs;
    if (clause.size() > 3) {

        size_t poz = clause.size() / 2;
        
        _kto3Cnfs.varNum ++;

        auto clauseA = Clause(clause.begin(), clause.begin() + poz);
        clauseA.emplace_back(_kto3Cnfs.varNum);
        auto clauseB = Clause(clause.begin() + poz, clause.end());
        clauseB.emplace_back(-_kto3Cnfs.varNum);

        if (clauseA.size() > 3 ) {
            auto newCnf = kSat23Sat(clauseA);
            tempCnfs.insert(tempCnfs.end(), newCnf.begin(), newCnf.end());
        } else {
            tempCnfs.push_back(clauseA);
        }

        if (clauseB.size() > 3) {
            auto newCnf = kSat23Sat(clauseB);
            tempCnfs.insert(tempCnfs.end(), newCnf.begin(), newCnf.end());
        } else {
            tempCnfs.push_back(clauseB);
        }
    } else if (clause.size() == 3) {
        tempCnfs.emplace_back(clause);
    }

    return tempCnfs;
}

void 
SatSolver::addAuxBitDim1(int var1, bool sign) 
{
    _QbitCnfs.varNum += 1;

    if (sign) {
        _QbitCnfs.clauses.emplace_back(Clause{_QbitCnfs.varNum, var1});
    } else {
        _QbitCnfs.clauses.emplace_back(Clause{-_QbitCnfs.varNum, var1});
    }
}

void 
SatSolver::addAuxBitDim3(int var1, int var2, int var3, bool sign)
{   
    _QbitCnfs.varNum += 1;
    if (sign) {
        _QbitCnfs.clauses.emplace_back(Clause{_QbitCnfs.varNum, var1});
        _QbitCnfs.clauses.emplace_back(Clause{_QbitCnfs.varNum, var2});
        _QbitCnfs.clauses.emplace_back(Clause{_QbitCnfs.varNum, var3});
    } else {
        _QbitCnfs.clauses.emplace_back(Clause{-_QbitCnfs.varNum, var1});
        _QbitCnfs.clauses.emplace_back(Clause{-_QbitCnfs.varNum, var2});
        _QbitCnfs.clauses.emplace_back(Clause{-_QbitCnfs.varNum, var3});
    }
}

void 
SatSolver::reportSol(Eigen::VectorXd &sol)
{
    size_t successed = 0;

    for (int i = 0; i < _Cnfs.clauseNum; ++i) {
        const auto clause = _Cnfs.clauses[i];
        bool result = false;
        for (const auto& literal : clause) {
            int poz = std::abs(literal) - 1;
            if ( (literal > 0 && sol(poz) > 0) || (literal < 0 && sol(poz) < 0)) {
                successed ++;
                result = true;
                break;
            }
        }
    }

    std::cout << "[Solution report] Total clauses: "<< _Cnfs.clauseNum << ", Successed clauses num: " << successed << ", Precent: "<< static_cast<double> (successed) / _Cnfs.clauseNum << std::endl;
}

void 
SatSolver::legalize(Eigen::VectorXd &sol) 
{
    size_t successed = 0;

    std::unordered_map<int, std::vector<int>> var2ClausesCone;   // lit <=> <clause idx>
    std::vector<int> failedClauses;

    for (int i = 0; i < _Cnfs.clauseNum; ++i) {
        for (int lit : _Cnfs.clauses[i]) {
            var2ClausesCone[std::abs(lit)].emplace_back(i);
        }
    }

    for (int i = 0; i < _Cnfs.clauseNum; ++i) {
        const auto clause = _Cnfs.clauses[i];
        bool result = false;
        for (const auto& literal : clause) {
            int poz = std::abs(literal) - 1;
            if ( (literal > 0 && sol(poz) > 0) || (literal < 0 && sol(poz) < 0)) {
                successed ++;
                result = true;
                break;
            }
        }

        if (!result) {
            failedClauses.push_back(i);
        }
    }

    // ty to fix failed
    for (const auto clauseIdx: failedClauses) {
        // check if its fixed by previous fix

        bool result = false;
        for (const auto& literal : _Cnfs.clauses[clauseIdx]) {
            int poz = std::abs(literal) - 1;
            if ( (literal > 0 && sol(poz) > 0) || (literal < 0 && sol(poz) < 0)) {
                result = true;
                break;
            }
        }

        if (result){
            successed ++;
            break;
        }

        for (const auto& literal : _Cnfs.clauses[clauseIdx]) {
            
            int poz = std::abs(literal) - 1;
            int oldValue = sol[poz];
            sol[poz] = -oldValue;   // invert var
            bool isValid = true;    // check other clause

            for (const auto& otherClauseIdx : var2ClausesCone[literal]) {
                if (otherClauseIdx == clauseIdx) continue;  // skip current
                const auto& otherClause = _Cnfs.clauses[otherClauseIdx];
                bool otherSatisfied = false;
                for (int otherLit : otherClause) {
                    int otherPoz = std::abs(otherLit) - 1;
                    if ((otherLit > 0 && sol[otherPoz] > 0) || (otherLit < 0 && sol[otherPoz] < 0)) {
                        otherSatisfied = true;
                        break;
                    }
                }
                if (!otherSatisfied) {
                    isValid = false;
                    break;
                }
            }

            // rollback
            if (!isValid) {
                sol[poz] = oldValue;
            } else {  // commit
                successed ++;
                break;
            }
        }
    }
    std::cout << "[Solution report] After fixing total clauses: "<< _Cnfs.clauseNum << ", Successed clauses num: " << successed << ", Precent: "<< static_cast<double> (successed) / _Cnfs.clauseNum << std::endl;

}

Eigen::MatrixXd
SatSolver::pyCubicOptimizerWrapper(const Eigen::Tensor<int, 3> &A, const Eigen::MatrixXd &B, const int dim)
{
    torch::Tensor tensorA = eigen_to_torch(A); 
    torch::Tensor tensorB = eigen_to_torch(B);

    py::scoped_interpreter guard{};

    py::module pyModule = py::module::import("qubic_optimizer");
    py::object pyClass = pyModule.attr("MyCubicOptimization");
    py::object cubicOptimizer = pyClass(
        tensorA,  // A
        tensorB,  // B
        dim       // dim
    );

    torch::Tensor hessian = cubicOptimizer.attr("hessian_at_origin")().cast<torch::Tensor>();
    std::cout << "Hessian:\n" << hessian << std::endl;

    // Convert the PyTorch tensor back to Eigen
    // return torch_to_eigen(hessian);

    Eigen::MatrixXd temp;
    return temp;
}

torch::Tensor 
SatSolver::eigen_to_torch(const Eigen::Tensor<int, 3>& e_tensor) {
    std::vector<int64_t> dims = {e_tensor.dimension(0), 
                              e_tensor.dimension(1), 
                              e_tensor.dimension(2)};
    
    std::vector<double> data(e_tensor.data(), 
                           e_tensor.data() + e_tensor.size());
    
    return torch::from_blob(data.data(), dims, torch::kFloat64).clone();
}

torch::Tensor 
SatSolver::eigen_to_torch(const Eigen::MatrixXd& eigen_mat) {

    std::vector<int64_t> dims = {eigen_mat.rows(), 
                                eigen_mat.cols()};

    std::vector<double> data(eigen_mat.data(), eigen_mat.data() + eigen_mat.size());
    
    return torch::from_blob(data.data(), dims, torch::kFloat64);
}

Eigen::Tensor<int, 3> 
SatSolver::torch_to_eigen(const torch::Tensor& t_tensor) {
    auto sizes = t_tensor.sizes();
    Eigen::Tensor<int, 3> e_tensor(sizes[0], sizes[1], sizes[2]);
    
    int* data = t_tensor.data_ptr<int>();
    
    std::copy(data, data + e_tensor.size(), e_tensor.data());
    
    return e_tensor;
}

void 
SatSolver::clean() 
{
    _currCnfPtr = nullptr;
    _Cnfs.clauses.clear();
    _kto3Cnfs.clauses.clear();
    _QbitCnfs.clauses.clear();
    _Cnfs.clauseNum = 0;
    _kto3Cnfs.clauseNum = 0;
    _QbitCnfs.clauseNum = 0;
    _Cnfs.varNum = 0;
    _kto3Cnfs.varNum = 0;
    _QbitCnfs.varNum = 0;
}   

}