#include <torch/torch.h>
#include <iostream>

class MyCubicOptimization {
public:
    MyCubicOptimization(torch::Tensor A, torch::Tensor B, int64_t dim) 
        : A_(A), B_(B), dim_(dim) {
        N_ = A.size(0);
    }

    int64_t N() const {
        return N_;
    }

    torch::Tensor objective(torch::Tensor sol) {
        TORCH_CHECK(sol.numel() == A_.size(0), "Solution size must match A's first dimension");
        auto _sol = sol.view({-1});
        
        // Cubic term: einsum("ijk,i,j,k", A, _sol, _sol, _sol)
        auto cubic_term = A_.mul_(_sol.unsqueeze(1).unsqueeze(2))
                          .mul_(_sol.unsqueeze(0).unsqueeze(2))
                          .mul_(_sol.unsqueeze(0).unsqueeze(1))
                          .sum();
        
        // Quadratic term: einsum("ij,i,j", B, _sol, _sol)
        auto quadratic_term = B_.mul_(_sol.unsqueeze(1))
                             .mul_(_sol.unsqueeze(0))
                             .sum();
        
        return cubic_term + quadratic_term;
    }

    torch::Tensor objective_no_tensor(torch::Tensor sol) {
        TORCH_CHECK(sol.numel() == A_.size(0), "Solution size must match A's first dimension");
        auto _sol = sol.view({-1});
        
        // Cubic term
        auto cubic_term = A_.mul_(_sol.unsqueeze(1).unsqueeze(2))
                          .mul_(_sol.unsqueeze(0).unsqueeze(2))
                          .mul_(_sol.unsqueeze(0).unsqueeze(1))
                          .sum();
        
        // Quadratic term
        auto quadratic_term = B_.mul_(_sol.unsqueeze(1))
                             .mul_(_sol.unsqueeze(0))
                             .sum();
        
        // No tensor term: einsum("i,j->", _sol, _sol)
        auto no_tensor_term = _sol.dot(_sol);
        
        return cubic_term + quadratic_term + no_tensor_term;
    }

    torch::Tensor to_state(torch::Tensor state) {
        return state;
    }

    torch::Tensor hessian_at_origin() {
        auto zeros = torch::zeros({dim_}, torch::kFloat32);
        
        // Lambda function for the objective
        auto obj_func = [this](torch::Tensor x) {
            return this->objective(x);
        };
        
        // Compute Hessian
        auto H = torch::autograd::hessian(obj_func, zeros);
        return H;
    }

    torch::Tensor hessian_at_origin_no_tensor() {
        auto zeros = torch::zeros({dim_}, torch::kFloat32);
        
        // Lambda function for the objective_no_tensor
        auto obj_func = [this](torch::Tensor x) {
            return this->objective_no_tensor(x);
        };
        
        // Compute Hessian
        auto H = torch::autograd::hessian(obj_func, zeros);
        return H;
    }

private:
    torch::Tensor A_;
    torch::Tensor B_;
    int64_t dim_;
    int64_t N_;
};

int main() {
    int64_t dim = 3;
    auto A = torch::randn({dim, dim, dim});
    auto B = torch::randn({dim, dim});
    
    MyCubicOptimization cubic_optimization(A, B, dim);
    
    std::cout << "Hessian at origin:\n" << cubic_optimization.hessian_at_origin() << std::endl;
    std::cout << "Hessian at origin (no tensor term):\n" 
              << cubic_optimization.hessian_at_origin_no_tensor() << std::endl;
    
    return 0;
}