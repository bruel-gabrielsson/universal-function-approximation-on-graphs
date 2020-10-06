#include <torch/extension.h>
#include "order.h"

#include <algorithm>
#include <iostream>
#include <cstdint>
using namespace std;

// if default == -2 then it will set as the index
// assume indices of size 2
void extend_if_needed_d1(std::vector<int> *vec_p, std::vector<int> *indices, int def) {
  // std::vector<int>::iterator result = std::max_element(indices->begin(), indices->end());
  size_t index;
  if (indices->size() > 1) {
    index = std::max((*indices)[0], (*indices)[1]); // index_p;
  } else {
    index = (*indices)[0];
  }

  size_t curr_size = vec_p->size();
  if (curr_size <= index) {
    for (size_t i = curr_size; i <= index; i++) {
      if ((int) def == -2) {
        vec_p->push_back((int) i);
      } else {
        vec_p->push_back(def);
      }
    }
  }
}

// if default == -2 then it will set as the index
void extend_if_needed_d2(std::vector<std::vector<int>> *vec_p, std::vector<int> *indices, int def) {
  size_t index = std::max((*indices)[0], (*indices)[1]);
  size_t curr_size = vec_p->size();
  if (curr_size <= index) {
    for (size_t i = curr_size; i <= index; i++) {
      if ((int) def == -2) {
        vec_p->push_back((std::vector<int> {(int) i}));
      } else {
        vec_p->push_back((std::vector<int> {(int) def}));
      }
    }
  }
}

// Custom compare for [[n1, n2, deg1, deg2, label1, label2], ...] edges
bool compareAll(std::vector<int> ar1, std::vector<int> ar2) {
  if (ar1[2]<ar2[2]) return 1;
  if (ar1[2]>ar2[2]) return 0;
  if (ar1[3]<ar2[3]) return 1;
  if (ar1[3]>ar2[3]) return 0;
  if (ar1[4]<ar2[4]) return 1;
  if (ar1[4]>ar2[4]) return 0;
  if (ar1[5]<ar2[5]) return 1;
  // if (ar1[5]>ar2[5]) return 0;
  return 0;
}

bool compareDegBoth(std::vector<int> ar1, std::vector<int> ar2) {
  if (ar1[2]<ar2[2]) return 1;
  if (ar1[2]>ar2[2]) return 0;
  if (ar1[3]<ar2[3]) return 1;
  // if (ar1[3]>ar2[3]) return 0;
  return 0;
}

bool compareDegOne(std::vector<int> ar1, std::vector<int> ar2) {
  if (ar1[2]<ar2[2]) return 1;
  // if (ar1[2]>ar2[2]) return 0;
  return 0;
}

// edges [[edgenum, n1, n2, deg1, deg2, label1, label2], ...]
// edges [[graph, edgenum, n1, n2]] Pass in so graphs appear consistently
// one pass to fix each graph (loop through every graph orderly)
// another pass to fix offset based on level lengths (level_to_offset array and just go through all edges of all graphs)
std::tuple< std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<int>> > preprocess_order(std::vector<std::vector<std::vector<int>>> vector_of_graph_edges, int option, bool powerful, int maxlevel) {
  std::vector<int> level_lengths;
  level_lengths.push_back(0);
  std::vector<std::vector<std::vector<int>>> all_processed_a;
  std::vector<std::vector<std::vector<int>>> all_processed_b;
  std::vector<std::vector<int>> all_processed_edges;
  for (size_t i = 0; i<vector_of_graph_edges.size(); i++) {
    std::vector<std::vector<int>> *vec = &vector_of_graph_edges[i];
    process_graph(vec, (int) i, &all_processed_edges, &level_lengths, option, powerful, maxlevel, &all_processed_a, &all_processed_b); // , &all_processed_b);
  }

  // print2d(&all_processed_a);
  // print3d(&all_processed_a);
  // cout << "Edn" << endl;
  // print3d(&all_processed_b);

  int accum = 0;
  std::vector<int> acc_level_lengths; //
  acc_level_lengths.push_back(0); // this is because we want to iterate later
  for (size_t i = 0; i < level_lengths.size(); i ++) {
    accum += level_lengths[i];
    acc_level_lengths.push_back(accum);
  }

  std::vector<std::vector<std::vector<int>>> levels; // from level to edges
  std::vector<std::vector<std::vector<int>>> levels_a;
  std::vector<std::vector<std::vector<int>>> levels_b;
  std::vector<std::vector<int>> sparseinds;
  int size_a = all_processed_a.size();
  if(size_a != (int) all_processed_b.size()) {
    cout << "ERROR size_a != (int) all_processed_b.size()" << endl;
  }
  for (size_t i = 0; i < all_processed_edges.size(); i++) {
    std::vector<int> edge_info = all_processed_edges[i];
    int level = edge_info[1] - 1; // [graph_num, level, level1, level2, offset1, offset2] -1 because we don't care about the 0, that's just all nodes
    int level1 = edge_info[2];
    int level2 = edge_info[3];
    int offset1 = edge_info[4];
    int offset2 = edge_info[5];
    int first = edge_info[6];
    int second = edge_info[7];
    int full_intersection = edge_info[8];
    if (level >= (int) levels.size()) {
      // std::vector<std::vector<int>> temp_level = {{acc_level_lengths[level1]+offset1, acc_level_lengths[level2]+offset2}};
      std::vector<std::vector<int>> temp_level = {{first, second, acc_level_lengths[level1]+offset1, acc_level_lengths[level2]+offset2, full_intersection}};
      levels.push_back(temp_level);
      // Always want oldest first!
      // if (level1 > level2) {
      //   std::vector<std::vector<int>> temp_level = {{acc_level_lengths[level1]+offset1, acc_level_lengths[level2]+offset2}};
      //   levels.push_back(temp_level);
      // } else if (level1 < level2) {
      //   std::vector<std::vector<int>> temp_level = {{acc_level_lengths[level2]+offset2, acc_level_lengths[level1]+offset1}};
      //   levels.push_back(temp_level);
      // } else { // if equal randomize
      //   if ( ((double) std::rand() / (RAND_MAX)) > 0.5 ) {
      //     std::vector<std::vector<int>> temp_level = {{acc_level_lengths[level1]+offset1, acc_level_lengths[level2]+offset2}};
      //     levels.push_back(temp_level);
      //   } else {
      //     std::vector<std::vector<int>> temp_level = {{acc_level_lengths[level2]+offset2, acc_level_lengths[level1]+offset1}};
      //     levels.push_back(temp_level);
      //   }
      // }

    } else {
      // levels[level].push_back({acc_level_lengths[level1]+offset1, acc_level_lengths[level2]+offset2});
      levels[level].push_back({first, second, acc_level_lengths[level1]+offset1, acc_level_lengths[level2]+offset2, full_intersection});
      // Always want oldest first!
      // if (level1 > level2) {
      //   levels[level].push_back({acc_level_lengths[level1]+offset1, acc_level_lengths[level2]+offset2});
      // } else if (level1 < level2) {
      //   levels[level].push_back({acc_level_lengths[level2]+offset2, acc_level_lengths[level1]+offset1});
      // } else { // if equal randomize
      //   if ( ((double) std::rand() / (RAND_MAX)) > 0.5 ) {
      //     levels[level].push_back({acc_level_lengths[level1]+offset1, acc_level_lengths[level2]+offset2});
      //   } else {
      //     levels[level].push_back({acc_level_lengths[level2]+offset2, acc_level_lengths[level1]+offset1});
      //   }
      // }
    }

    if (powerful) {
      std::vector<std::vector<int>> this_a = all_processed_a[i];
      int a_size = this_a.size();
      if (a_size == 0) {
        if (level >= (int) levels_a.size()) {
          levels_a.push_back({});
        }
      } else {
        for (int k = 0; k < a_size; k++) {
          // {graph_num, level, (*level_lengths_so_far_p)[0] + node, (*level_lengths_so_far_p)[level] - 1}
          std::vector<int> node_info = this_a[k];
          int level = node_info[1];
          int node_offset = node_info[2];
          int a_level_offset = node_info[3];
          if (level >= (int) levels_a.size()) {
            levels_a.push_back({{ node_offset, acc_level_lengths[level] + a_level_offset }});
          } else {
            levels_a[level].push_back({{ node_offset, acc_level_lengths[level] + a_level_offset }});
          }
        }
      }

      std::vector<std::vector<int>> this_b = all_processed_b[i];
      int b_size = this_b.size();
      if (b_size == 0) {
        if (level >= (int) levels_b.size()) {
          levels_b.push_back({});
        }
      } else {
        for (int k = 0; k < b_size; k++) {
            // {graph_num, level, (*level_lengths_so_far_p)[0] + node, (*level_lengths_so_far_p)[level] - 1}
            std::vector<int> node_info = this_b[k];
            int level = node_info[1];
            int node_offset = node_info[2];
            int b_level_offset = node_info[3];
            if (level >= (int) levels_b.size()) {
              levels_b.push_back({{ node_offset, acc_level_lengths[level] + b_level_offset }});
            } else {
              levels_b[level].push_back({{ node_offset, acc_level_lengths[level] + b_level_offset }});
            }
        }
      }
    }

    sparseinds.push_back({edge_info[0], acc_level_lengths[level+1]-acc_level_lengths[1] + ((int)levels[level].size())-1});
  }

  std::vector<torch::Tensor> tensor_levels;
  std::vector<torch::Tensor> levels_a_tensor;
  std::vector<torch::Tensor> levels_b_tensor;

  //tensor_levels = (vector<torch::Tensor> *) &levels;
  int ind_size = 5; // 2
  for (size_t ii = 0; ii < levels.size(); ii++) {
    std::vector<std::vector<int>> this_level = levels[ii];
    auto tensor = torch::empty((int) this_level.size() * ind_size, torch::TensorOptions().dtype(torch::kInt64));
    int64_t* data = tensor.data<int64_t>();
    for (const auto& i : this_level) {
        for (const auto& j : i) {
            *data++ = j;
        }
    }
    tensor_levels.push_back(tensor.resize_({(int) this_level.size(), ind_size})); // torch::tensor(levels[i]), torch::requires_grad(false)); // .dtype(torch::kInt32)); //, {levels[i].size(), 2});

    if (powerful) {
      std::vector<std::vector<int>> a_this_level = levels_a[ii];
      auto tensor_a = torch::empty((int) a_this_level.size() * 2, torch::TensorOptions().dtype(torch::kInt64));
      int64_t* data_a = tensor_a.data<int64_t>();
      for (const auto& i : a_this_level) {
          for (const auto& j : i) {
              *data_a++ = j;
          }
      }
      levels_a_tensor.push_back(tensor_a.resize_({(int) a_this_level.size(), 2}));

      std::vector<std::vector<int>> b_this_level = levels_b[ii];
      auto tensor_b = torch::empty((int) b_this_level.size() * 2, torch::TensorOptions().dtype(torch::kInt64));
      int64_t* data_b = tensor_b.data<int64_t>();
      for (const auto& i : b_this_level) {
          for (const auto& j : i) {
              *data_b++ = j;
          }
      }
      levels_b_tensor.push_back(tensor_b.resize_({(int) b_this_level.size(), 2}));
    }
  }

  std::tuple< std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<int>> > to_return = std::make_tuple(tensor_levels, levels_a_tensor, levels_b_tensor, sparseinds);
  return to_return;
}

// Will shuffle and sort edges and return ordered edges
// as well as num_levels, levels_size, num_nodes
void process_graph(std::vector<std::vector<int>> *edges_p, int graph_num, std::vector<std::vector<int>> *processed_edges_p, std::vector<int> *level_lengths_so_far_p, int option, bool powerful, int maxlevel,
    std::vector<std::vector<std::vector<int>>> *all_processed_a_p, std::vector<std::vector<std::vector<int>>> *all_processed_b_p) { //}, std::vector<std::vector<int>> *all_processed_b_p) {

  std::random_shuffle(edges_p->begin(), edges_p->end()); // Shuffle before sort
  if (option == 0) std::stable_sort(edges_p->begin(), edges_p->end(), compareDegOne);
  if (option == 1) std::stable_sort(edges_p->begin(), edges_p->end(), compareDegBoth);
  if (option == 2) std::stable_sort(edges_p->begin(), edges_p->end(), compareAll);
  // it's sorted now!
  // the algorithm!
  // check size to see if should add or not
  std::vector<int> node_to_group_num;
  std::vector<std::vector<int>> group_num_to_group;
  std::vector<int> group_num_to_level;
  std::vector<int> group_num_to_offset;
  int max_node_num = 0;
  for (size_t i = 0; i < edges_p->size(); i++) {
    std::vector<int> edge = (*edges_p)[i];
    int first = edge[0], second = edge[1];
    if ( ((double) std::rand() / (RAND_MAX)) > 0.5 ) { // we randomize order ATM
      first = second;
      second = edge[0];
    }

    max_node_num = std::max({first,second,max_node_num});
    std::vector<int> edge_nodes = {first, second};
    extend_if_needed_d1(&node_to_group_num, &edge_nodes, -2);
    extend_if_needed_d2(&group_num_to_group, &edge_nodes, -2);
    int group_num1 = node_to_group_num[first], group_num2 = node_to_group_num[second];
    std::vector<int> group_nums = {group_num1, group_num2};
    // if (group_num1 == group_num2) continue; // OBS, just a test!!!!! Avoids loops
    extend_if_needed_d1(&group_num_to_level, &group_nums, 0);
    extend_if_needed_d1(&group_num_to_offset, &group_nums, -2);
    int level1 = group_num_to_level[group_num1], level2 = group_num_to_level[group_num2];

    // if (option == 2 && level1 < level2) {
    //   int temp = first;
    //   first = second;
    //   second = temp;
    //   temp = group_num1;
    //   group_num1 = group_num2;
    //   group_num2 = temp;
    //   temp = level1;
    //   level1 = level2;
    //   level2 = temp;
    // }

    int level = std::max(level1, level2) + 1;
    if (maxlevel > 0 && level > maxlevel) continue;
    int offset1 = group_num_to_offset[group_num1], offset2 = group_num_to_offset[group_num2];
    if (level1 == 0) offset1 += (*level_lengths_so_far_p)[0];
    if (level2 == 0) offset2 += (*level_lengths_so_far_p)[0];
    int full_intersection = (int) (group_num1 == group_num2);
    processed_edges_p->push_back({graph_num, level, level1, level2, offset1, offset2, (*level_lengths_so_far_p)[0] + first, (*level_lengths_so_far_p)[0] + second, full_intersection});
    // all_processed_a_p->push_back({});
    std::vector<int> temp_level_vec = {level};
    extend_if_needed_d1(level_lengths_so_far_p, &temp_level_vec, 0);

    // Now update values
    group_num_to_level[group_num1] = level;
    group_num_to_offset[group_num1] = (*level_lengths_so_far_p)[level]; // starting with zero so
    (*level_lengths_so_far_p)[level] += 1; // incrementing level for this one, has to be done after we use it
    // if (group_num1 == group_num2) continue; // else we will update same group and increase forever
    std::vector<std::vector<int>> list_a;
    std::vector<std::vector<int>> list_b;
    if (powerful && (group_num1 != group_num2)) { // Obs, must be done before we change the groups
        for (size_t j = 0; j < group_num_to_group[group_num1].size(); j++) {
          int nodea = (int) group_num_to_group[group_num1][j];
          list_a.push_back({graph_num, level-1, (*level_lengths_so_far_p)[0] + nodea, (*level_lengths_so_far_p)[level] - 1});
        }
    }

    if (group_num1 != group_num2) {
      for (size_t j = 0; j < group_num_to_group[group_num2].size(); j++) { // copying group2 onto group1
        int node = (int) group_num_to_group[group_num2][j];

        if (powerful) {
          // this should point to the c value that was just created, -1 because we incremented it above, level-1 because how we iterate in implementation
          list_b.push_back({graph_num, level-1, (*level_lengths_so_far_p)[0] + node, (*level_lengths_so_far_p)[level] - 1});
        }

        if (group_num1 != group_num2) {
          group_num_to_group[group_num1].push_back(node);
          node_to_group_num[node] = group_num1;
        }
      }
    }

    // cout << list_a.size() << endl;
    if (powerful) {
      all_processed_b_p->push_back(list_b);
      all_processed_a_p->push_back(list_a);
    }
  }
  (*level_lengths_so_far_p)[0] += max_node_num + 1;
}

void print2d(std::vector<std::vector<int>> *vec_p) {
  int height = vec_p->size(), width = (*vec_p)[0].size();
  for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
          std::cout << (*vec_p)[i][j] << ' ';
      }
      std::cout << std::endl;
  }
}

void print3d(std::vector<std::vector<std::vector<int>>> *vec_p) {
  int height = vec_p->size(); // , width = (*vec_p)[0].size();
  for (int i = 0; i < height; ++i) {
      for (int j = 0; j < (int)(*vec_p)[i].size(); ++j) {
          std::cout << "[" << ' ';
          for (int k = 0; k < (int)(*vec_p)[i][j].size(); ++k) {
            std::cout << (*vec_p)[i][j][k] << ' ';
          }
          std::cout << ']';
      }
      std::cout << std::endl;
  }
}
