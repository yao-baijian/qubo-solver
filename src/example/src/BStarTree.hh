#include <vector>
#include <queue>
#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>

#include "define.hh"
#include "BsbSolver.hh"

namespace Floorplan {

class BStarTree
{

public:
    BStarTree(float dead_space_ratio, Sb::BsbSolver *bsbSolver) 
        : dead_space_ratio(dead_space_ratio),
        area_normal(0), 
        wl_normal(0), 
        root(-1), 
        feasible(false),
        _bsbSolver(bsbSolver){};

    ~BStarTree() = default;

    void parse_hardblocks(std::string inputfile);
    void parse_net(std::string inputfile);
    void parse_pl(std::string inputfile);
    void initBstarTree();
    
    bool SimulatedAnnealing();
    
    void initSimulatedBifurcation(size_t trialNum, int numIters, double dT);
    bool puboBsb();
    bool quboBsb(const std::string &mode);

    void write_output(std::string output_file);

    // getter
    size_t getBlocksNum() { return num_hard; }
    
    std::vector<Terminal*> terminals;
    std::vector<Block*> hardblocks;
    std::vector<std::vector<int>> nets;
    std::vector<Block*> min_cost_floorplan, Fixedmin_cost_floorplan;
    std::vector<Node*> min_cost_bstar, Fixedmin_cost_bstar;
    int num_hard, num_terminal, num_net, num_pin;

    Eigen::MatrixXd weightMatrix;


    // floorplan
    int W;      // fixed outline max x coordinate
    double dead_space_ratio;
    // double area_normal, wl_normal;
    int root;
    std::vector<Node*> bstar_tree;
    std::vector<int> contour;
    bool in_fixed_outline;
    int min_cost_root, Fixedmin_cost_root;      // floorplan with min cost ,in fixed outline
    Cost min_cost, Fixedmin_cost;
    bool feasible;

    double area_normal, wl_normal;

private:
    void rotate(int node);
    void move(int from, int to);
    void swap(int n1, int n2);

    Cost cost();
    Cost dynaTerminalCost();
    double evalHpwl(const std::vector<int> &net);
    double evalHpwlNoTerm(const std::vector<int> &net);

    void preorder_traverse(int node, bool left);
    void updateCenterCoordi();
    void updateArea();

    // floorplan
    int _fpWidth, _fpHeight;
    double _fpArea, _whRatio;

    // SA
    const int _timeLimit = 40;
    size_t _trialNum;

    Sb::BsbSolver *_bsbSolver;
};

}