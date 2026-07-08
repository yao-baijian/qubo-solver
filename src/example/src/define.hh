#pragma once

#include <vector>
#include <set>
#include <map>

#define SEED 1542959801

namespace Sat {

    using Clause = std::vector<int>;

    struct CNF {
        int varNum = 0;
        int clauseNum = 0;
        std::vector<Clause> clauses;
    };

    enum class MappingMode {
        AUXILIARY,
        HESSIAN,
    };

    typedef struct satOptions {
        MappingMode mappingMode;
        bool batch;
        size_t quboReduceMode;
        bool fix;
        double hessianCoeff;
        double penaltyCoeff;
    } SatOptions;

}

namespace Sb {
    enum class SolverMode {
        CPU,
        GPU,
    };

    typedef struct sbSolverOptions {
        SolverMode solverMode;
        Sat::MappingMode mapping;
        size_t iterNum;
        double dt;
    } SbSolverOptions;
}

namespace Floorplan {

using NodeId = size_t;
using Net = std::vector<int>;
using Nets = std::vector<std::vector<int>>;

typedef struct info{
    double total;
    double max;
    double min;
    size_t num;
} Info; 

enum class Orient {
    VERTICAL,
    HORIZONTAL,
};

class Block {
public:
    Block() : 
        id(0), 
        x(0), 
        y(0), 
        width(0), 
        height(0), 
        orient(Orient::HORIZONTAL) {};

    Block(int id, int x, int y, int width, int height, Orient ori) : 
        id(id), 
        x(x), 
        y(y), 
        width(width), 
        height(height), 
        orient(ori) {};

    ~Block () = default;

    // getter
    size_t getArea() const { return width * height; }    
    Nets getNet() const { return _nets; };

    // setter
    void addNet(std::vector<int> net) { _nets.emplace_back(net);};

    // logger
    void logNet() const {
        for (auto net : _nets) {
            std::cout << "<";
            for (auto inst : net) {
                std::cout << "sb" << inst << " ";
            }
            std::cout << "> ";
        }
        std::cout << std::endl;
    }

    int id;
    int x;
    int y;
    int width;
    int height;
    Orient orient;

private:
    Nets _nets;
};

class Terminal {

public:
    Terminal() : 
        id(0), 
        x(0), 
        y(0) {};

    Terminal(int id, int x, int y) : 
        id(id), 
        x(x), 
        y(y) {};

    ~Terminal () = default;

    int id;
    int x;
    int y;
};

// for tree
class Node {
public:
    Node() : 
        parent(-1), 
        self(-1), 
        leftchild(-1), 
        rightchild(-1) {};

    Node(int parent, NodeId self, int leftchild, int rightchild) : 
        parent(parent), 
        self(self),
        leftchild(leftchild), 
        rightchild(rightchild) {};

    ~Node () = default;

    int parent;
    NodeId self;
    int leftchild;
    int rightchild;
};

// for floorplan
class Cost {

public:
    Cost() : 
        _area(0), 
        _hpwl(0), 
        _aspectRatio(0), 
        _width(0), 
        _height(0), 
        _cost(0) {};

    Cost(double area, double hpwl, double aspectRatio, int width, int height) : 
        _area(area), 
        _hpwl(hpwl), 
        _aspectRatio(aspectRatio), 
        _width(width), 
        _height(height), 
        _cost(0) {};

    ~Cost () = default;

    // calc
    void calcost1(double initArea, double initHpwl, int size) {
        double width_penalty = 0, height_penalty = 0;
        if(_width > size)
            width_penalty = (double) _width / size;
        if(_height > size)
            height_penalty = (double) _height / size;

        _cost = (_area/initArea) + (_hpwl / initHpwl) + 0.6* pow(1 - _aspectRatio, 2) + width_penalty + height_penalty;
    };

    // getter
    double getHpwl() const {return _hpwl;};
    double getArea() const {return _area;};
    int getWidth() const {return _width;};
    int getHeight() const {return _height;};
    double getCost() const {return _cost;};
    double getRatio() const {return _aspectRatio;};

    // logger
    void log() const {
        printf("[       Width      ] : %d\n", _width);
        printf("[       Height     ] : %d\n", _height);
        printf("[       Area       ] : %.0f\n",  _area);
        printf("[     Wirelength   ] : %.0f\n", _hpwl);
        printf("[        R         ] : %f\n", _aspectRatio);
        printf("[       Cost       ] : %f\n\n\n", _cost);
    }

private:
    double _area;
    double _hpwl;
    double _aspectRatio;   // aspect ratio
    int _width;
    int _height;
    double _cost;
};
}