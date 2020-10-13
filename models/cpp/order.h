#include <torch/extension.h>
#include <vector>

void extend_if_needed_d1(std::vector<int> *vec_p, std::vector<size_t> *indices, int def);
void extend_if_needed_d2(std::vector<std::vector<int>> *vec_p, std::vector<int> *indices, int def);

bool compareAll(std::vector<int> ar1, std::vector<int> ar2);
bool compareDegBoth(std::vector<int> ar1, std::vector<int> ar2);
bool compareDegOne(std::vector<int> ar1, std::vector<int> ar2);

void print2d(std::vector<std::vector<int>> *vec_p);
void print3d(std::vector<std::vector<std::vector<int>>> *vec_p);

//std::vector<torch::Tensor> preprocess_order(std::vector<torch::Tensor> vector_of_graph_edges);
std::tuple< std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<int>> > preprocess_order(std::vector<std::vector<std::vector<int>>> vector_of_graph_edges, int option, bool powerful, int maxlevel, bool sort_nodes);

// int process_graph(torch::Tensor edges);
void process_graph(std::vector<std::vector<int>> *edges_p, int graph_num, std::vector<std::vector<int>> *processed_edges_p, std::vector<int> *level_lengths_so_far_p, int option, bool powerful, bool sort_nodes, int maxlevel,
  std::vector<std::vector<std::vector<int>>> *all_processed_a_p, std::vector<std::vector<std::vector<int>>> *all_processed_b_p); // , std::vector<std::vector<int>> *all_processed_b_p);
