#ifndef ALGEBRASOLVER_HH
#define ALGEBRASOLVER_HH

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <set>

namespace Sat {
    
class AlgebraicSolver {

public:

    std::map<std::vector<int>, int> solve(const Clause inputs) {
        std::vector<int> coefficients(8, 0);

        if (inputs.size() > 3) {
            throw std::invalid_argument("Algebraic solver only supports up to 3 variables.");
        }

        std::vector<int> coeff;

        if (inputs.size() == 1) { coeff = solve1(inputs);
        } else if (inputs.size() == 2) { coeff = solve2(inputs);
        } else if (inputs.size() == 3) { coeff = solve3(inputs);}

        std::map<std::vector<int>, int> coeffTerm;
        for (size_t i = 0; i < term_order.size(); ++i) {
            if (coeff[i] == 0) continue; // Skip zero coefficients
            coeffTerm.emplace(term_order[i], coeff[i]);
        }

        return coeffTerm;
    }

private:

    std::vector<std::vector<int>> term_order;

    std::vector<int> solve1(const Clause& inputs) {
        std::vector<int> coefficients(8, 0);
        // TODO
        return coefficients;
    }

    std::vector<int> solve2(const Clause& inputs) {

        term2(inputs);
        std::vector<int> coefficients(4, 0);

        auto expanded_terms = generateExpandedTerms(inputs);
        
        for (const auto& term : expanded_terms) {
            coefficients[termToIndex(term)] += (term.size() % 2 == 0) ? 1 : -1;
        }
        return coefficients;
    }

    std::vector<int> solve3(const Clause& inputs) {

        term3(inputs);
        std::vector<int> coefficients(8, 0);
        
        auto expanded_terms = generateExpandedTerms(inputs);
        
        for (const auto& term : expanded_terms) {
            coefficients[termToIndex(term)] += (term.size() % 2 == 0) ? 1 : -1;
        }
        
        return coefficients;
    }

    void term2(const Clause& inputs){
        term_order.clear();
        const int i1 = abs(inputs[0]), i2 = abs(inputs[1]);
        term_order = { {}, {i1}, {i2}, {i1, i2}};

    }

    void term3(const Clause& inputs){

        term_order.clear();
        const int i1 = abs(inputs[0]), i2 = abs(inputs[1]), i3 = abs(inputs[2]);
        term_order = {{}, {i1}, {i2}, {i3}, {i1, i2}, {i2, i3}, {i3, i1}, {i1, i2, i3}};
    }

    int termToIndex(const std::vector<int>& term) const {
        auto sorted_term = term;
        std::sort(sorted_term.begin(), sorted_term.end());

            
        for (size_t i = 0; i < term_order.size(); ++i) {
            auto sorted_pattern = term_order[i];
            std::sort(sorted_pattern.begin(), sorted_pattern.end());
            
            if (sorted_pattern == sorted_term) {
                return static_cast<int>(i);
            }
        }

        std::cerr << "Error: Term not found in term_order. Term: ";
        for (auto v : term) std::cerr << v << " ";
        std::cerr << std::endl;
        throw std::runtime_error("Term not found in term_order");
    }

    std::vector<std::vector<int>> generateExpandedTerms(const Clause& inputs) const {
        std::vector<std::vector<std::vector<int>>> factors;
        

        for (const auto& inp : inputs) {
            int var_index = abs(inp);
            
            if (inp > 0) { 
                factors.push_back({{var_index}});
            } else {       
                factors.push_back({{}, {var_index}}); // 1 - xi
            }
        }
        
        std::vector<std::vector<int>> expanded_terms;
        generateProductTerms(factors, 0, {}, expanded_terms);
        return expanded_terms;
    }

    void generateProductTerms(const std::vector<std::vector<std::vector<int>>>& factors, 
                            size_t depth, 
                            std::vector<int> current, 
                            std::vector<std::vector<int>>& result) const {
        if (depth == factors.size()) {

            std::set<int> unique_vars(current.begin(), current.end());
            std::vector<int> term(unique_vars.begin(), unique_vars.end());
            result.push_back(term);
            return;
        }
        
        for (const auto& factor : factors[depth]) {
            auto new_current = current;
            new_current.insert(new_current.end(), factor.begin(), factor.end());
            generateProductTerms(factors, depth + 1, new_current, result);
        }
    }
};

}

#endif
