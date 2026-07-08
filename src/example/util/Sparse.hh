#include <vector>

struct SparseTensor3D {
    std::vector<std::tuple<int, int, int, int>> entries; // (i,j,k,value)
    int dim1, dim2, dim3; // Dimensions of the tensor
};