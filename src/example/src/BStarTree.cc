#include "BStarTree.hh"
#include <Eigen/Sparse>
#include <algorithm>

namespace Floorplan {

void 
BStarTree::parse_hardblocks(std::string inputfile)
{   
    std::ifstream file;
    file.open(inputfile);

    std::string Num, colon, str;
    file >> Num >> colon >> num_hard;       //  NumHardRectilinearBlocks : number of hard rectilinear block nodes
    file >> Num >> colon >> num_terminal;   //  NumTerminals : number of terminal (pad etc.) nodes
    file >> str;                            //  nodeName hardrectilinear vertexNumber vertex1, vertex2, ..., vertexN
    
    //  ex : sb0 hardrectilinear 4 (0, 0) (0, 33) (43, 33) (43, 0)
    hardblocks = std::vector<Block*>(num_hard);
    double total_block_area = 0;    // A = total block area
    for (int i = 0; i < num_hard; i++) {
        getline(file, str);

        size_t left_paren = str.find("(");
        left_paren = str.find("(", left_paren + 1);
        left_paren = str.find("(", left_paren + 1);
        size_t comma = str.find(",");
        comma = str.find(",", comma + 1);
        comma = str.find(",", comma + 1);
        size_t right_paren = str.find(")");
        right_paren = str.find(")", right_paren + 1);
        right_paren = str.find(")", right_paren + 1);

        char buffer[100];
        int width, height;
        size_t len = str.copy(buffer, comma-left_paren-1, left_paren+1);
        buffer[len] = '\0';
        width = atoi(buffer);
        len = str.copy(buffer, right_paren-comma-2, comma+2);
        buffer[len] = '\0';
        height = atoi(buffer);

        hardblocks[i] = new Block(i, -1, -1, width, height, Orient::HORIZONTAL);
        total_block_area += hardblocks[i]->getArea();
    }

    //H*W* = total block area * (1 + dead space ratio)
    double fixed_outline_area = total_block_area * (1 + dead_space_ratio);
    W = sqrt(fixed_outline_area);

    printf("\nFixed-outline constraint info \n");
    printf("[ Floorplan Region ] : %.2f\n", fixed_outline_area);
    printf("[ Total Block Area ] : %.2f\n", total_block_area);
    printf("[ Fixed-Outline Width ] : %d\n\n", W);

    file.close();
}

void
BStarTree::parse_net(std::string inputfile)
{ 
    std::ifstream file;
    file.open(inputfile);

    std::string Num, colon, line;
    file >> Num >> colon >> num_net;
    file >> Num >> colon >> num_pin;
    std::cout << Num << std::endl;

    nets = std::vector<std::vector<int>>(num_net);

    weightMatrix = Eigen::MatrixXd::Zero(num_hard, num_hard);

    for (int i=0; i<num_net ; i++){

        Net net;
        int degree;
        file >> Num >> colon >> degree;

        for(int j=0; j < degree ; j++){
            file >> line;
            int id;
            if(line[0]=='p'){
                line.erase(0,1);
                id = atoi(line.c_str()) + num_hard;
            }
            else if(line[0]=='s'){
                line.erase(0,2);
                id = atoi(line.c_str());
                net.emplace_back(id);
            }
            nets[i].emplace_back(id);
        }
        
        if (net.size() <= 1){
            continue;
        }

        // add nets 2 block
        // TODO add topology manager
        for(int j=0; j<net.size() ; j++){
            hardblocks[net[j]]->addNet(net);
        }

        // transfer weight matrix to a upper matrix

        std::sort(net.begin(), net.end());
        auto headBlock = net[0];
        net.erase(net.begin());
        
        for(int j=0; j<net.size() ; j++){
            weightMatrix(headBlock, net[j]) += 1;
        }
    }
    file.close();
}

void
BStarTree::parse_pl(std::string inputfile)
{
    terminals = std::vector<Terminal*>(num_terminal + 1); 
    std::ifstream file;
    file.open(inputfile);
    std::string nodeName;
    int x, y;
    for(int i=1 ; i<=num_terminal ; i++){
        file >> nodeName >> x >> y;     //nodeName XY-coordinate
        terminals[i] =  new Terminal(i, x, y);
    }
    file.close();
}

void 
BStarTree::initBstarTree()
{
    bstar_tree = std::vector<Node*>(num_hard);

    for (size_t i = 0; i < bstar_tree.size(); ++i) {
        bstar_tree[i] = new Node();
    }
    
    std::queue<int> bfs;
    std::vector<int> insert(num_hard,0);

    root = rand() % num_hard;
    bstar_tree[root]->parent = -1;
    bstar_tree[root]->self = root;
    bfs.push(root);
    insert[root] = 1;

    int i = num_hard-1;
    while (!bfs.empty()){
        int parent = bfs.front();
        int left_child = -1, right_child = -1;
        bfs.pop();

        if(i > 0){
            do{
                left_child = rand() % num_hard;
            } while (insert[left_child]);
            bstar_tree[parent]->leftchild = left_child;
            bstar_tree[left_child]->self = left_child;
            bstar_tree[left_child]->parent = parent;

            bfs.push(left_child);
            insert[left_child] = 1;
            i--;
        }

        if(i > 0){
            do{
                right_child = rand() % num_hard;
            } while (insert[right_child]);
            bstar_tree[parent]->rightchild = right_child;
            bstar_tree[right_child]->self = right_child;
            bstar_tree[right_child]->parent = parent;

            bfs.push(right_child);
            insert[right_child] = 1;
            i--;
        }

        bstar_tree[parent]->leftchild = left_child;
        bstar_tree[parent]->rightchild = right_child;
    }
            
}

void
BStarTree::rotate(int node)
{
    int temp = hardblocks[node]->width;
    hardblocks[node]->width = hardblocks[node]->height;
    hardblocks[node]->height = temp;

    if (hardblocks[node]->orient == Orient::HORIZONTAL)
        hardblocks[node]->orient = Orient::VERTICAL;
    else
        hardblocks[node]->orient = Orient::HORIZONTAL;
}

void 
BStarTree::move(int from, int to)
{
    /* delete the node */
    // no child
    if(bstar_tree[from]->leftchild == -1 && bstar_tree[from]->rightchild == -1){
        int parent = bstar_tree[from]->parent;
        if(bstar_tree[parent]->leftchild == from)
            bstar_tree[parent]->leftchild = -1;
        else if(bstar_tree[parent]->rightchild == from)
            bstar_tree[parent]->rightchild = -1;
        else{
            printf("[Error] perturb : moving !\n");
            exit(1);
        }
    }
    // two child
    else if(bstar_tree[from]->leftchild != -1 && bstar_tree[from]->rightchild != -1){
        do{
            bool move_left;
            if(bstar_tree[from]->leftchild != -1 && bstar_tree[from]->rightchild != -1)
                move_left = rand() % 2 == 0;
            else if(bstar_tree[from]->leftchild != -1)
                move_left = true;
            else
                move_left = false;
            
            if(move_left)
                swap(from, bstar_tree[from]->leftchild);
            else
                swap(from, bstar_tree[from]->rightchild);
        }while(bstar_tree[from]->leftchild != -1 || bstar_tree[from]->rightchild != -1);

        int parent = bstar_tree[from]->parent;
        if(bstar_tree[parent]->leftchild == from)
            bstar_tree[parent]->leftchild = -1;
        else if(bstar_tree[parent]->rightchild == from)
            bstar_tree[parent]->rightchild = -1;
        else{
            printf("[Error] perturb : moving !\n");
            exit(1);
        }
    }
    // one child
    else{
        int child;
        if(bstar_tree[from]->leftchild != -1)
            child = bstar_tree[from]->leftchild;
        else
            child = bstar_tree[from]->rightchild;

        int parent = bstar_tree[from]->parent;
        if(parent != -1){
            if(bstar_tree[parent]->leftchild == from)
                bstar_tree[parent]->leftchild = child;
            else if(bstar_tree[parent]->rightchild == from)
                bstar_tree[parent]->rightchild = child;
            else{
                printf("[Error] perturb : moving !\n");
                exit(1);
            }
        }
        bstar_tree[child]->parent = parent;
        
        //root may change
        if(from == root)
            root = child;
    }
    
    /* insert the node */
    int insert_pos = rand() % 4;
    int child;
    if(insert_pos == 0){
        child = bstar_tree[to]->leftchild;
        bstar_tree[from]->leftchild = child;
        bstar_tree[from]->rightchild = -1;
        bstar_tree[to]->leftchild = from;
    }
    else if(insert_pos == 0){
        child = bstar_tree[to]->rightchild;
        bstar_tree[from]->leftchild = child;
        bstar_tree[from]->rightchild = -1;
        bstar_tree[to]->rightchild = from;
    }
    else if(insert_pos == 0){
        child = bstar_tree[to]->leftchild;
        bstar_tree[from]->leftchild = -1;
        bstar_tree[from]->rightchild = child;
        bstar_tree[to]->leftchild = from;
    }
    else{
        child = bstar_tree[to]->rightchild;
        bstar_tree[from]->leftchild = -1;
        bstar_tree[from]->rightchild = child;
        bstar_tree[to]->rightchild = from;
    }
    bstar_tree[from]->parent = to;
    if(child != -1)
        bstar_tree[child]->parent = from; 
}

void 
BStarTree::swap(int n1, int n2)
{
    /* swap two parent */
    int n1_parent = bstar_tree[n1]->parent;
    if(n1_parent != -1){
        if(bstar_tree[n1_parent]->leftchild == n1)
            bstar_tree[n1_parent]->leftchild = n2;
        else if(bstar_tree[n1_parent]->rightchild == n1)
            bstar_tree[n1_parent]->rightchild = n2;
        else{
            printf("[Error] perturb : swapping !\n");
            exit(1);
        }
    }

    int n2_parent = bstar_tree[n2]->parent;
    if(n2_parent != -1){
        if(bstar_tree[n2_parent]->leftchild == n2)
            bstar_tree[n2_parent]->leftchild = n1;
        else if(bstar_tree[n2_parent]->rightchild == n2)
            bstar_tree[n2_parent]->rightchild = n1;
        else{
            printf("[Error] perturb : swapping !\n");
            exit(1);
        }
    }

    bstar_tree[n1]->parent = n2_parent;
    bstar_tree[n2]->parent = n1_parent;

    /* swap two child */
    int n1_leftchild = bstar_tree[n1]->leftchild;
    int n1_rightchild = bstar_tree[n1]->rightchild;
    bstar_tree[n1]->leftchild = bstar_tree[n2]->leftchild;
    bstar_tree[n1]->rightchild = bstar_tree[n2]->rightchild;
    bstar_tree[n2]->leftchild = n1_leftchild;
    bstar_tree[n2]->rightchild = n1_rightchild;

    if(bstar_tree[n1]->leftchild != -1)
        bstar_tree[bstar_tree[n1]->leftchild]->parent = n1;
    if(bstar_tree[n1]->rightchild != -1)
        bstar_tree[bstar_tree[n1]->rightchild]->parent = n1;
    if(bstar_tree[n2]->leftchild != -1)
        bstar_tree[bstar_tree[n2]->leftchild]->parent = n2;
    if(bstar_tree[n2]->rightchild != -1)
        bstar_tree[bstar_tree[n2]->rightchild]->parent = n2;
    
    /* swap parent & child */
    if(bstar_tree[n1]->parent == n1)
        bstar_tree[n1]->parent = n2;
    else if(bstar_tree[n1]->leftchild == n1)
        bstar_tree[n1]->leftchild = n2;
    else if(bstar_tree[n1]->rightchild == n1)
        bstar_tree[n1]->rightchild = n2;

    if(bstar_tree[n2]->parent == n2)
        bstar_tree[n2]->parent = n1;
    else if(bstar_tree[n2]->leftchild == n2)
        bstar_tree[n2]->leftchild = n1;
    else if(bstar_tree[n2]->rightchild == n2)
        bstar_tree[n2]->rightchild = n1;
    
    // root may change
    if(n1 == root)
        root = n2;
    else if(n2 == root)
        root = n1;
}

void 
BStarTree::preorder_traverse(int node, bool left){
    int parent = bstar_tree[node]->parent;

     if(left)
         hardblocks[node]->x = hardblocks[parent]->x + hardblocks[parent]->width;
     else
        hardblocks[node]->x = hardblocks[parent]->x;

    // mantain contour
    int x_left = hardblocks[node]->x;
    int x_right = x_left + hardblocks[node]->width;
    int y_max = 0;
    for(int i=x_left; i<x_right ; i++){
        if(contour[i] > y_max){
            y_max = contour[i];
        }
    }
    hardblocks[node]->y = y_max;
    y_max += hardblocks[node]->height;
    for (int i=x_left; i<x_right ; i++){
        contour[i] = y_max;
    }

    if(bstar_tree[node]->leftchild != -1)
        preorder_traverse(bstar_tree[node]->leftchild, true);
    if(bstar_tree[node]->rightchild != -1)
        preorder_traverse(bstar_tree[node]->rightchild, false);
}

void 
BStarTree::updateCenterCoordi(){
    contour = std::vector<int>(5*W,0);
    hardblocks[root]->x = 0;
    hardblocks[root]->y = 0;
    for(int i=0; i<hardblocks[root]->width; i++)
        contour[i] = hardblocks[root]->height;
    
    if(bstar_tree[root]->leftchild != -1)
        preorder_traverse(bstar_tree[root]->leftchild, true);
    if(bstar_tree[root]->rightchild != -1)
        preorder_traverse(bstar_tree[root]->rightchild, false);
}

void 
BStarTree::updateArea(){
    _fpWidth = 0;
    _fpHeight = 0;
    for(int i=0 ; i < num_hard ; i++){
        if(hardblocks[i]->x + hardblocks[i]->width > _fpWidth) {
            _fpWidth = hardblocks[i]->x + hardblocks[i]->width;
        }
        if(hardblocks[i]->y + hardblocks[i]->height > _fpHeight) {
            _fpHeight = hardblocks[i]->y + hardblocks[i]->height;
        }
    }
    _fpArea = _fpWidth * _fpHeight;
    _whRatio = (double) _fpHeight / _fpWidth;
}

Cost 
BStarTree::cost(){
    double hpwl = 0;
    for(const std::vector<int> &net : nets){
        hpwl += evalHpwl(net);
    }

    auto cost = Cost (_fpArea, hpwl, _whRatio, _fpWidth, _fpHeight);

    if(area_normal == 0)
        area_normal = _fpArea;
    if(wl_normal == 0)
        wl_normal = hpwl;

    cost.calcost1(area_normal, wl_normal, W);
    
    return cost;
}

// TODO: add fast hpwl eval
double 
BStarTree::evalHpwl(const std::vector<int> &net) {
    int x_min = _fpWidth + 1;
    int x_max = 0;
    int y_min = _fpHeight + 1;
    int y_max = 0;
    for(const int pin : net){
        if(pin < num_hard) {
            int x_center = hardblocks[pin]->x + hardblocks[pin]->width/2;
            int y_center = hardblocks[pin]->y + hardblocks[pin]->height/2;
            if(x_center < x_min)
                x_min = x_center;
            if(y_center < y_min)
                y_min = y_center;
            if(x_center > x_max)
                x_max = x_center;
            if(y_center > y_max)
                y_max = y_center;
        } else {
            auto pinPtr = terminals[pin-num_hard];
            if(pinPtr->x < x_min)
                x_min = pinPtr->x;
            if(pinPtr->y < y_min)
                y_min = pinPtr->y;
            if(pinPtr->x > x_max)
                x_max = pinPtr->x;
            if(pinPtr->y > y_max)
                y_max = pinPtr->y;
        }
    }
    
    return (x_max - x_min) + (y_max - y_min);
}

double 
BStarTree::evalHpwlNoTerm(const std::vector<int> &net) {
    int x_min = _fpWidth + 1;
    int x_max = 0;
    int y_min = _fpHeight + 1;
    int y_max = 0;
    for(const int pin : net){
        if(pin < num_hard) {
            int x_center = hardblocks[pin]->x + hardblocks[pin]->width/2;
            int y_center = hardblocks[pin]->y + hardblocks[pin]->height/2;
            if(x_center < x_min)
                x_min = x_center;
            if(y_center < y_min)
                y_min = y_center;
            if(x_center > x_max)
                x_max = x_center;
            if(y_center > y_max)
                y_max = y_center;
        }
    }
    
    return (x_max - x_min) + (y_max - y_min);
}

Cost 
BStarTree::dynaTerminalCost(){
    double hpwl = 0;
    for(const auto net : nets){
        hpwl += evalHpwlNoTerm(net);
    }

    auto cost = Cost (_fpArea, hpwl, _whRatio, _fpWidth, _fpHeight);

    if(area_normal == 0)
        area_normal = _fpArea;
    if(wl_normal == 0)
        wl_normal = hpwl;
    
    cost.calcost1(area_normal, wl_normal, W);
    
    return cost;
}

bool 
BStarTree::SimulatedAnnealing(){
    
    updateCenterCoordi();
    updateArea();
    min_cost = cost();

    min_cost_floorplan = hardblocks;
    const double P = 0.95;
    const double r = 0.9;
    const int k = 20;
    const int N = k * num_hard;
    const double T0 = -min_cost.getCost() * num_hard / log(P);

    double T = T0;
    int MT = 0;     // try次數
    int uphill = 0;
    int reject = 0;
    Cost old_cost = min_cost;
    in_fixed_outline = false;

    clock_t time_start = clock();
    clock_t time = time_start;
    const int max_seconds = (num_hard/20) * (num_hard/20);
    int seconds = 0, runtime = 0;

    do{
        MT = 0;
        uphill = 0;
        reject = 0;

        do{
            std::vector<Block*> temp_hard(hardblocks);
            std::vector<Node*> temp_bstar(bstar_tree);
            int old_root = root;

            /* Perturbing */
            int perturb = rand() % 3;
            // M1 : rotate
            if(perturb == 0){
                int node = rand() % num_hard;
                rotate(node);
            }
            // M2 : swap
            else if(perturb == 1){
                int n1,n2;
                n1 = rand() % num_hard;
                do{
                    n2 = rand() % num_hard;
                } while(n2==n1);
                swap(n1,n2);
            }
            // M3 : move
            else if(perturb == 2){
                int from, to;
                from = rand() % num_hard;
                do{
                    to = rand() % num_hard;
                } while(to==from || bstar_tree[from]->parent==to);
                move(from,to);
            }

            MT++;
            updateCenterCoordi();
            updateArea();
            Cost new_cost = cost();
            double delta_cost = new_cost.getCost() - old_cost.getCost();
            double random = (double)rand() / RAND_MAX;
            if(delta_cost<=0 || random<exp(-delta_cost/T)){     // p21. line 17
                if(delta_cost > 0)
                    uphill++;
                
                // feasible solution with minimum cost
                if(new_cost.getWidth()<=W && new_cost.getHeight()<=W){
                    if(in_fixed_outline){
                        if(new_cost.getCost() < Fixedmin_cost.getCost()){
                            Fixedmin_cost_root = root;
                            Fixedmin_cost = new_cost;
                            Fixedmin_cost_floorplan = hardblocks;
                            Fixedmin_cost_bstar = bstar_tree;
                        }
                    }
                    else{
                        in_fixed_outline = true;
                        Fixedmin_cost_root = root;
                        Fixedmin_cost = new_cost;
                        Fixedmin_cost_floorplan = hardblocks;
                        Fixedmin_cost_bstar = bstar_tree;
                    }
                }
                // infeasible solution with minimum cost
                if(new_cost.getCost() < min_cost.getCost()){
                    min_cost_root = root;
                    min_cost = new_cost;
                    min_cost_floorplan = hardblocks;
                    min_cost_bstar = bstar_tree;
                }

                old_cost = new_cost;
            }
            else{
                reject++;
                root = old_root;
                if(perturb == 0)
                    hardblocks = temp_hard;
                else
                    bstar_tree = temp_bstar;
            }
        } while(uphill<=N && MT<=2*N);

        T *= r;

        seconds = (clock()-time) / CLOCKS_PER_SEC;
        runtime = (clock()-time_start) / CLOCKS_PER_SEC;
        if(seconds>=max_seconds && in_fixed_outline==false){
            printf("Overtime %d %d\n",min_cost.getWidth(), min_cost.getHeight());
            seconds = 0;
            time = clock();
            T = T0;
        }
    } while(seconds<max_seconds && runtime < _timeLimit);

    if(in_fixed_outline){
        feasible = 1;
        printf("\nFound feasible solution\n");
        min_cost.log();
    }
    else{
        feasible = 0;
        printf("Cannot found feasible solution\n");
        return false;
    }
    return true;
}

void 
BStarTree::initSimulatedBifurcation(size_t trialNum, int numIters, double dT) {
    _trialNum = trialNum;
}

bool 
BStarTree::puboBsb() {
    return true;
}

/**
 *  qubo by swap pair
 *  O(i,j) = weight(i,j) * d(i,j)
 *  d(i,j) = 1 - (si * sj) (to be decided)
 *  weight(i,j) = \coe1 * hpwl_diff(i,j) + \coe2 * sqrt (area_diff(i,j)) + \coe3 * width_penalty + \coe4 * height_penalty
 *  hpwl_diff(i,j) = (hpwl_after(i,j) - hpwl_before(i,j)) 
 *  area_diff(i,j) = (area_after(i,j) - area_before(i,j)
 */

bool 
BStarTree::quboBsb(const std::string &mode)
{
    updateCenterCoordi();
    updateArea();
    min_cost = dynaTerminalCost();
    min_cost.log();
    min_cost_floorplan = hardblocks;

    size_t dim = weightMatrix.rows();
    size_t currTrial = 0;

    Cost curr_cost;
    while (currTrial < _trialNum && !in_fixed_outline) {
        std::cout << "bsb run: " << currTrial << std::endl;
        currTrial ++;

        Eigen::VectorXd spins (dim);
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
        std::vector<Eigen::MatrixXd> weight (4);
        std::vector<Eigen::MatrixXd> penalty (1);

        for (size_t i = 0; i < weight.size(); ++i) {
            weight[i] = Eigen::MatrixXd::Zero(dim, dim);
        }

        for (int i = 0; i < dim; ++i) {
            for (int j = i + 1; j < dim; ++j) {
                auto widthPenCoeff = 2.0, heightPenCoeff = 2.0;
                auto fromNet = hardblocks[i]->getNet();
                auto toNet = hardblocks[j]->getNet();
                // TODO: exclude same net here
                // TODO: how to connect wire weight matrix with hwpl ?
                double areaBefore = 0.0;
                int widthBefore = 0, heightBefore = 0;

                for(const auto net : fromNet){ weight[0](i, j) += evalHpwlNoTerm(net);}
                for(const auto net : toNet){ weight[0](i, j) += evalHpwlNoTerm(net);}
                
                areaBefore = _fpArea;
                widthBefore = _fpWidth;
                heightBefore = _fpHeight;

                auto testCost1 = dynaTerminalCost();

                swap(i, j);
                updateCenterCoordi();
                updateArea();

                for(const auto net : fromNet){ weight[0](i, j) -= evalHpwlNoTerm(net);}
                for(const auto net : toNet){ weight[0](i, j) -= evalHpwlNoTerm(net); }

                auto areaDiff =  _fpArea - areaBefore;
                if (areaDiff < 0) {
                    areaDiff = -sqrt(abs(areaDiff));
                } else {
                    areaDiff = sqrt(areaDiff);
                }

                weight[1](i, j) = areaDiff;

                if (_fpWidth <= W && widthBefore <= W) { widthPenCoeff = 0.1;}
                if (_fpHeight <= W && heightBefore <= W) { heightPenCoeff = 0.1;}

                weight[2](i, j) = (_fpWidth - widthBefore) * widthPenCoeff;
                weight[3](i, j) = (_fpHeight - heightBefore) * heightPenCoeff;

                auto testCost2 = dynaTerminalCost();

                // swap back
                swap(i, j);
                updateCenterCoordi();
                updateArea();

                auto testCost3 = dynaTerminalCost();

            }
        }

        Eigen::VectorXd sol;

        auto penaltyCoeff = -0.1;
        auto weightCoeff = 1;

        // constraint the penalty
        // Sum(e(i,j)) = 1 when j = [i+1, dim] 
        // rewrite as Min: (1 - Sum(e(i,j))) ^ 2
        Eigen::MatrixXd Penalty = Eigen::MatrixXd::Zero(dim*dim, dim*dim);
        for (int i = 0; i < dim*dim; ++i) {
            for (int j = 0; j < dim*dim; ++j) {
                if (i == j) {continue;}
                Penalty(i, j) = 1.0;
            }
        }

        if (mode == "sparse") {
            // to N^2
            Eigen::SparseMatrix <double> sparseH(dim*dim, dim*dim);
            for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < dim; ++j) {
                    sparseH.insert(i*dim + j, i*dim + j) = H(i, j);
                }
            }
            sparseH.makeCompressed();
            sol = _bsbSolver->solve(sparseH);
        } else {
            _bsbSolver->genH(H, weight);
            Eigen::MatrixXd denseH = Eigen::MatrixXd::Zero(dim*dim, dim*dim);
            for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < dim; ++j) {
                    denseH(i*dim + j, i*dim + j) = H(i, j);
                }
            }
            Eigen::MatrixXd newH = denseH * weightCoeff + Penalty * penaltyCoeff;
            std::cout << "\ntransformed hamitonian [10 x 10]:\n" << newH.topLeftCorner(10, 10) << std::endl;
            
            Eigen::Tensor<int, 3> A (1, 1, 1);

            sol = _bsbSolver->solve(newH, false, A);
        }
        
        // legalize the solution
        std::cout << "legalizing...." << std::endl;
        auto solMap = _bsbSolver->legalize(H, sol, dim);
        

        // perform solution
        // swap
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                if (solMap(i, j) > 0) {
                    swap(i, j);
                }
            }
        }
    
        // rotation 
        for (int i = 0; i < num_hard; ++i) {
            rotate(i);
            updateCenterCoordi();
            updateArea();
            auto rotateCost = dynaTerminalCost();
            if (rotateCost.getCost() > min_cost.getCost()) {
                rotate(i);
            }
        }

        updateCenterCoordi();
        updateArea();

        curr_cost = dynaTerminalCost();
        curr_cost.log();
    
        if(curr_cost.getCost() < min_cost.getCost()){
            min_cost_root = root;
            min_cost = curr_cost;
            min_cost_floorplan = hardblocks;
            min_cost_bstar = bstar_tree;
    
            if(curr_cost.getWidth() <= W && curr_cost.getHeight() <= W){
                if( in_fixed_outline ){
                    if(curr_cost.getCost() < Fixedmin_cost.getCost()){
                        Fixedmin_cost_root = root;
                        Fixedmin_cost = curr_cost;
                        Fixedmin_cost_floorplan = hardblocks;
                        Fixedmin_cost_bstar = bstar_tree;
                    }
                }
                else{
                    in_fixed_outline = true;
                    Fixedmin_cost_root = root;
                    Fixedmin_cost = curr_cost;
                    Fixedmin_cost_floorplan = hardblocks;
                    Fixedmin_cost_bstar = bstar_tree;
                }
            }
        }
    }
    
    if(in_fixed_outline){
        feasible = 1;
        printf("\nFound feasible solution\n");
        min_cost.log();
    }
    else{
        feasible = 0;
        printf("Cannot found feasible solution\n");
        return false;
    }
    return true;
}

void 
BStarTree::write_output(std::string output_file)
{
    std::ofstream file;
    file.open(output_file);

    file << "Wirelength " << Fixedmin_cost.getHpwl() << "\n";
    file << "Blocks\n";

    for(int i=0; i<num_hard ; i++){
        file << "sb" << i << " " << Fixedmin_cost_floorplan[i]->x << " " << Fixedmin_cost_floorplan[i]->y << "\n";
    }

    file.close();
}

}