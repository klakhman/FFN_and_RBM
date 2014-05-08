#include "FFN.h"
#include "Data.h"
#include "RBM.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <utility>
#include <random>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;


FFN::FFN(unsigned int nLayers, vector<unsigned int> neuronsDistr) {
  activations_.resize(nLayers);
  weights_.resize(nLayers - 1);
  weights_grads_.resize(nLayers - 1);
  local_grads_.resize(nLayers - 1);
  momentum_speed_.resize(nLayers - 1);
  for (unsigned int layer = 0; layer < nLayers; ++layer)
    activations_[layer] = cv::Mat::zeros(neuronsDistr[layer] + 1, 1, CV_64F);
  for (unsigned int layer = 0; layer < nLayers - 1; ++layer) {
    weights_[layer] = cv::Mat::zeros(neuronsDistr[layer + 1], neuronsDistr[layer] + 1, CV_64F);
    weights_grads_[layer] = cv::Mat::zeros(neuronsDistr[layer + 1], neuronsDistr[layer] + 1, CV_64F);
    momentum_speed_[layer] = cv::Mat::zeros(neuronsDistr[layer + 1], neuronsDistr[layer] + 1, CV_64F);
    local_grads_[layer] = cv::Mat::zeros(neuronsDistr[layer + 1], 1, CV_64F);
  }
  out_layer_type_ = SIGMOID;
}

FFN::FFN(const RBM& rbm, vector<unsigned int> out_layers_neurons_distr) {
  const unsigned int nLayers = 2 + static_cast<unsigned int>(out_layers_neurons_distr.size());
  activations_.resize(nLayers);
  weights_.resize(nLayers - 1);
  weights_grads_.resize(nLayers - 1);
  local_grads_.resize(nLayers - 1);
  momentum_speed_.resize(nLayers - 1);

  std::vector<unsigned int> neuronsDistr{ static_cast<unsigned int>(rbm.weights_.cols),
    static_cast<unsigned int>(rbm.weights_.rows) };

  for (auto& neurons_n : out_layers_neurons_distr)
    neuronsDistr.push_back(neurons_n);

  for (unsigned int layer = 0; layer < nLayers; ++layer)
    activations_[layer] = cv::Mat::zeros(neuronsDistr[layer] + 1, 1, CV_64F);

  //cv::Mat rand_biases(neuronsDistr[1], 1, CV_64F);
  //randu(rand_biases, cv::Scalar::all(0), cv::Scalar::all(0));
  hconcat(rbm.hid_biases_.clone(), rbm.weights_.clone(), weights_[0]);

  for (unsigned int layer = 0; layer < nLayers - 1; ++layer) {
    if (layer != 0) weights_[layer] = cv::Mat::zeros(neuronsDistr[layer + 1], neuronsDistr[layer] + 1, CV_64F);
    weights_grads_[layer] = cv::Mat::zeros(neuronsDistr[layer + 1], neuronsDistr[layer] + 1, CV_64F);
    momentum_speed_[layer] = cv::Mat::zeros(neuronsDistr[layer + 1], neuronsDistr[layer] + 1, CV_64F);
    local_grads_[layer] = cv::Mat::zeros(neuronsDistr[layer + 1], 1, CV_64F);
  }
  out_layer_type_ = SIGMOID;
}

FFN::FFN(const vector<RBM>& rbms, vector<unsigned int> out_layers_neurons_distr) {
  const unsigned int nLayers = static_cast<unsigned int>(rbms.size()) + 1 + static_cast<unsigned int>(out_layers_neurons_distr.size());
  activations_.resize(nLayers);
  weights_.resize(nLayers - 1);
  weights_grads_.resize(nLayers - 1);
  local_grads_.resize(nLayers - 1);
  momentum_speed_.resize(nLayers - 1);

  std::vector<unsigned int> neuronsDistr; 
  neuronsDistr.push_back(rbms.front().weights_.cols);
  
  for (const auto& rbm : rbms)
    neuronsDistr.push_back(rbm.weights_.rows);

  for (auto& neurons_n : out_layers_neurons_distr)
    neuronsDistr.push_back(neurons_n);

  for (unsigned int layer = 0; layer < nLayers; ++layer)
    activations_[layer] = cv::Mat::zeros(neuronsDistr[layer] + 1, 1, CV_64F);

  //cv::Mat rand_biases(neuronsDistr[1], 1, CV_64F);
  //randu(rand_biases, cv::Scalar::all(0), cv::Scalar::all(0));
  for (unsigned int layer = 0; layer < rbms.size(); ++layer)
    hconcat(rbms[layer].hid_biases_.clone(), rbms[layer].weights_.clone(), weights_[layer]);

  for (unsigned int layer = 0; layer < nLayers - 1; ++layer) {
    if (layer >= rbms.size()) weights_[layer] = cv::Mat::zeros(neuronsDistr[layer + 1], neuronsDistr[layer] + 1, CV_64F);
    weights_grads_[layer] = cv::Mat::zeros(neuronsDistr[layer + 1], neuronsDistr[layer] + 1, CV_64F);
    momentum_speed_[layer] = cv::Mat::zeros(neuronsDistr[layer + 1], neuronsDistr[layer] + 1, CV_64F);
    local_grads_[layer] = cv::Mat::zeros(neuronsDistr[layer + 1], 1, CV_64F);
  }
  out_layer_type_ = SIGMOID;
}

Mat FFN::calculate(const Mat& inputs) {
  forward_pass(inputs);
  Mat outputs;
  activations_.back().rowRange(1, activations_.back().rows).copyTo(outputs);
  return outputs;
}


pair<vector<double>, vector<double> > FFN::train(unsigned int nIter, LearnParam learn_param, const Data& train_data_raw, const Data& val_data /*= Data()*/) {
  learn_param_ = learn_param;

  unsigned int mini_batch_n;
  const Data& train_data = (learn_param.mini_batch_size == 0) ? train_data_raw :
                          reorder_train_data(train_data_raw);
  
  mini_batch_n = (learn_param.mini_batch_size == 0) ? 1 :
    train_data.get_inputs().cols / learn_param.mini_batch_size + 
    ((train_data.get_inputs().cols % learn_param.mini_batch_size) > 0);

  const Mat& inputs = train_data.get_inputs();
  const Mat& targets = train_data.get_targets();

  vector<double> train_cost(nIter);
  vector<double> val_cost(nIter);

  for (unsigned int layer = 0; layer < momentum_speed_.size(); ++layer)
    momentum_speed_[layer] = 0;
  for (unsigned int iter = 0; iter < nIter; ++iter) {
    for (unsigned int mini_batch = 0; mini_batch < mini_batch_n; ++mini_batch) {
      Range samples_range;
      if (learn_param.mini_batch_size == 0)
        samples_range = Range::all();
      else
        samples_range = 
          Range(mini_batch * learn_param.mini_batch_size,
                min(static_cast<unsigned int>(inputs.cols), (mini_batch + 1) * learn_param.mini_batch_size));

      const Mat& batch_inputs = inputs.rowRange(Range::all()).colRange(samples_range);
      const Mat& batch_targets = targets.rowRange(Range::all()).colRange(samples_range);

      forward_pass(batch_inputs);
      backward_pass(batch_targets);
      compute_weights_grads();

      //numerical_check_grads(train_data);

      update_weigth();
    }

    train_cost[iter] = cost(train_data, true);
    if (val_data.get_inputs().cols != 0)
      val_cost[iter] = cost(val_data, true);
    
    cout << iter << ": " << train_cost[iter] << "\t" << val_cost[iter] << endl;
  }

  cout << endl << "Classification error: train_data: " << classification_error(train_data)
    << "; val_data: " << ((val_data.get_inputs().cols != 0) ? classification_error(val_data) : 0.0) << endl;

  return make_pair(train_cost, val_cost);
}

Data FFN::reorder_train_data(const Data& train_data_raw) {
  const Mat& inputs = train_data_raw.get_inputs();
  const Mat& targets = train_data_raw.get_targets();

  Data reordered_train_data;
  Mat& reordered_inputs = reordered_train_data.get_inputs_io();
  reordered_inputs = Mat(inputs.rows, inputs.cols, CV_64F);
  Mat& reordered_targets = reordered_train_data.get_targets_io();
  reordered_targets = Mat(targets.rows, targets.cols, CV_64F);

  vector<unsigned int> indices(inputs.cols);
  unsigned int id = 0;
  for (auto& index : indices)
    index = id++;
  shuffle(begin(indices), end(indices), mt19937{ static_cast<unsigned int>(time(0)) });

  for (unsigned int index = 0; index < indices.size(); ++index) {
    inputs.col(indices[index]).copyTo(reordered_inputs.col(index));
    targets.col(indices[index]).copyTo(reordered_targets.col(index));
  }
  return reordered_train_data;
}

void FFN::rand_weights(double range) {
  //for (unsigned int layer = 0; layer < nLayers() - 1; ++layer)
  //  randu(weights_[layer], Scalar::all(-range), Scalar::all(range));
  for (unsigned int layer = 1; layer <= nLayers() - 1; ++layer)
    rand_weights(range, layer);
}

void FFN::rand_weights(double range, unsigned int layer_n) {
  for (int row = 0; row < weights_[layer_n - 1].rows; ++row)
  for (int column = 0; column < weights_[layer_n - 1].cols; ++column)
    weights_[layer_n - 1].at<double>(row, column) = rand_unif_real(-range, range);
  //randu(weights_[layer_n - 1], Scalar::all(-range), Scalar::all(range));
}

double FFN::rand_unif_real(double A, double B) {
  static mt19937 re{static_cast<unsigned int>(time(0))};
  using Dist = uniform_real_distribution<double>;
  static Dist uid{};
  return uid(re, Dist::param_type(A, B));
}

int FFN::rand_unif_int(int A, int B) {
  static mt19937 re{ static_cast<unsigned int>(time(0)) };
  using Dist = uniform_int_distribution<int>;
  static Dist uid{};
  return uid(re, Dist::param_type(A, B));
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
    case CV_8U: r = "8U"; break;
    case CV_8S: r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default: r = "User"; break;
  }

  r + "C";
  r += (chans + '0');

  return r;
}

void FFN::forward_pass(const Mat& input) {
  weights_[0].type();
  vconcat(Mat(1, input.cols, CV_64F, 1.0), input, activations_.front());

  for (unsigned int layer = 1; layer < activations_.size() - 1; ++layer) {
    //cout << type2str(weights_[layer - 1].type()) << endl;
    //cout << type2str(activations_[layer - 1].type()) << endl;
    //cout << type2str(activations_[layer].type()) << endl;
    //cout << endl << endl;;
    vconcat(Mat(1, activations_[layer - 1].cols, CV_64F, 1.0),
      logistic(weights_[layer - 1] * activations_[layer - 1]), activations_[layer]);
  }

  calculate_out_layer();
}

void FFN::calculate_out_layer() {
  switch (out_layer_type_) {
    case SIGMOID:
      vconcat(Mat(1, activations_[nLayers() - 2].cols, CV_64F, 1.0),
        logistic(weights_[nLayers() - 2] * activations_[nLayers() - 2]), activations_.back());
      break;

    case SOFTMAX:
      // Считаем softmax вычислительно устойчивым способом
      Mat potentials = weights_[nLayers() - 2] * activations_[nLayers() - 2];

      Mat maxs; reduce(potentials, maxs, 0, CV_REDUCE_MAX);
      Mat tmp;
      exp(potentials - repeat(maxs, potentials.rows, 1), tmp);
      reduce(tmp, tmp, CV_REDUCE_SUM, 0);
      log(tmp, tmp);
      Mat normalizer = tmp + maxs;

      Mat log_class_prob = potentials - repeat(normalizer, potentials.rows, 1);
      Mat class_prob; exp(log_class_prob, class_prob);

      vconcat(Mat(1, activations_[nLayers() - 2].cols, CV_64F, 1.0),
        class_prob, activations_.back());
      break;
  }  
}

void FFN::backward_pass(const Mat& target) {
  compute_fin_layer_grads(target);

  for (unsigned int layer = nLayers() - 1; layer >= 2; --layer)
    compute_local_grads(layer);
}

void FFN::compute_fin_layer_grads(const Mat& target) {
  Mat row_activations = activations_.back().rowRange(1, activations_.back().rows);
  switch (out_layer_type_) {
    case SIGMOID:
      // dE/dy
      local_grads_.back() = row_activations - target;
      // dE/dz
      multiply(local_grads_.back(), row_activations, local_grads_.back());
      multiply(local_grads_.back(), 1 - row_activations, local_grads_.back());
      break;
    case SOFTMAX:
      local_grads_.back() = row_activations - target;
      break;
  }
}

void FFN::compute_local_grads(unsigned int nLayer) {
  Mat tmp_grad = (weights_[nLayer - 1].t() * local_grads_[nLayer - 1]);
  // dE/dy
  local_grads_[nLayer - 2] = tmp_grad.rowRange(1, tmp_grad.rows);
  // dE/dz
  Mat row_activations = activations_[nLayer - 1].rowRange(1, activations_[nLayer - 1].rows);
  multiply(local_grads_[nLayer - 2], row_activations, local_grads_[nLayer - 2]);
  multiply(local_grads_[nLayer - 2], 1 - row_activations, local_grads_[nLayer - 2]);
}

void FFN::compute_weights_grads() {
  for (unsigned int layer = nLayers() - 1; layer > 0; --layer) {
    const unsigned int data_size = local_grads_[layer - 1].cols;
    hconcat(Mat::zeros(weights_[layer - 1].rows, 1, CV_64F), 
      learn_param_.weight_decay_coef * weights_[layer - 1].colRange(1, weights_[layer - 1].cols), weights_grads_[layer - 1]);
    weights_grads_[layer - 1] += (1.0 / data_size) * local_grads_[layer - 1] * activations_[layer - 1].t();
  }
}

void FFN::update_weigth() {
  for (unsigned int layer = nLayers() - 1; layer > 0; --layer) {
    momentum_speed_[layer - 1] = learn_param_.momentum * momentum_speed_[layer - 1] + weights_grads_[layer - 1];
    weights_[layer - 1] -= learn_param_.learn_rate * momentum_speed_[layer - 1];
  }
}

//void FFN::update_weigth_Nesterov_pre() {
//
//}
//
//void FFN::update_weigth_Nesterov_post() {
//
//}

void FFN::numerical_check_grads(const Data& data) {
  const double delta_w = 1e-2;
  const double eps = 1e-5;
  const vector<double> contribution_distances{ -4, -3, -2, -1, 1, 2, 3, 4 };
  const vector<double> contribution_weights{ 1.0/280, -4.0/105, 1.0/5, -4.0/5, 4.0/5, -1.0/5, 4.0/105, -1.0/280 };

  for (unsigned int layer = 0; layer < nLayers() - 1; ++layer)
    for (unsigned int trial = 0; trial < 40; ++trial) {
      unsigned int pre_node = static_cast<unsigned int>(rand_unif_int(0, weights_[layer].cols - 1));
      unsigned int post_node = static_cast<unsigned int>(rand_unif_int(0, weights_[layer].rows - 1));
      double numerical_grad = 0;
      for (unsigned int step = 0; step < contribution_distances.size(); ++step) {
        weights_[layer].at<double>(post_node, pre_node) += delta_w * contribution_distances[step];
        numerical_grad += cost(data) * contribution_weights[step];
        weights_[layer].at<double>(post_node, pre_node) -= delta_w * contribution_distances[step];
      }
      numerical_grad /= delta_w;
      if (abs(weights_grads_[layer].at<double>(post_node, pre_node) - numerical_grad) > eps) {
        cout << endl << "Gradient checking failed: weight's gradient at layer " << layer + 1
          << " from neuron " << pre_node << " to neuron " << post_node << "." << endl
          << "Analytical gradient: " << weights_grads_[layer].at<double>(post_node, pre_node)
          << ", Numerical gradient: " << numerical_grad;
        exit(1);
      }
    }
}

double FFN::cost(const Data& data, bool raw /*= false*/) {
  const Mat& inputs = data.get_inputs();
  forward_pass(inputs);

  return cost_const(data, raw); 
}

double FFN::cost_const(const Data& data, bool raw /*= false*/) {
  const Mat& targets = data.get_targets();
  double class_error = 0;
 
  switch (out_layer_type_) {
    case SIGMOID: {
      Mat error_mat;
      error_mat = activations_.back().rowRange(1, activations_.back().rows) - targets;
      pow(error_mat, 2, error_mat);
      class_error = (1.0 / (2 * targets.cols)) * sum(error_mat)[0];
      break;
    }
    case SOFTMAX:
      Mat tmp;
      log(activations_.back().rowRange(1, activations_.back().rows), tmp);
      multiply(tmp, targets, tmp);
      class_error = -sum(tmp)[0] / targets.cols;
      break;
  }


  double weight_sum = 0;
  if (!raw)
    for (unsigned int layer = 0; layer < weights_.size(); ++layer) {
      Mat weight_sqr;
      pow(weights_[layer].colRange(1, weights_[layer].rows), 2, weight_sqr);
      weight_sum += sum(weight_sqr)[0];
    }

  double cost = class_error + learn_param_.weight_decay_coef * weight_sum / 2;

  return cost;
}

double FFN::classification_error(const Data& data) {
  const Mat& inputs = data.get_inputs();
  const Mat& targets = data.get_targets();
  forward_pass(inputs);

  double error_rate = 0;
  for (int sample = 0; sample < inputs.cols; ++sample) {
    Point estimation, target;
    minMaxLoc(activations_.back().rowRange(1, activations_.back().rows).col(sample), 
      NULL, NULL, NULL, &estimation);
    minMaxLoc(targets.col(sample),
      NULL, NULL, NULL, &target);
    if (estimation.y != target.y) error_rate += 1;
  }

  return error_rate / inputs.cols;
}

Mat FFN::logistic(const Mat& z) {
  Mat res;
  exp(-z, res);
  return 1 / (1 + res);
}

Mat FFN::imagesc(const Mat& source) {
  double min;
  double max;
  minMaxIdx(source, &min, &max, NULL, NULL);

  unsigned int nRows = source.rows;
  unsigned int nCols = source.cols;

  Mat source_sc(nRows, nCols, CV_8UC1);

  if (source.isContinuous()) {
    nCols *= nRows;
    nRows = 1;
  }

  for (unsigned int i = 0; i < nRows; ++i) {
    uchar* d = source_sc.ptr<uchar>(i);
    const double* p = source.ptr<double>(i);
    for (unsigned int j = 0; j < nCols; ++j)
      d[j] = static_cast<uchar>((p[j] - min) / (max - min) * 255);
  }

  return source_sc;
}

void FFN::visualize_first_layer() {
  unsigned int n_hid = weights_[0].rows;
  unsigned int n_rows = static_cast<unsigned int>(ceil(sqrt(n_hid) + 0.5));

  unsigned int mini_image_size = static_cast<unsigned int>(sqrt(weights_[0].cols - 1) + 0.5);

  unsigned int vertical_dist = 4;
  unsigned int horizontal_dist = 4;

  Mat image = Mat::zeros(n_rows * mini_image_size + (n_rows + 1) * horizontal_dist,
                          n_rows * mini_image_size + (n_rows + 1) * vertical_dist, CV_64F);

  unsigned int row = 0;
  unsigned int col = 0;

  for (unsigned int node = 0; node < n_hid; ++node) {
    Range cols_range = Range(horizontal_dist * (col + 1) + mini_image_size * col,
                            horizontal_dist * (col + 1) + mini_image_size * (col + 1));
    Range rows_range = Range(vertical_dist * (row + 1) + mini_image_size * row,
                            vertical_dist * (row + 1) + mini_image_size * (row + 1));
    Mat mini_image = weights_[0].row(node)
                    .colRange(1, static_cast<unsigned int>(pow(mini_image_size, 2) + 0.5) + 1);
    mini_image = mini_image.reshape(0, mini_image_size);
    Mat mini_image_pos = image(rows_range, cols_range);

    mini_image.copyTo(mini_image_pos);
    col = (++col) % n_rows;
    row += (col % n_rows == 0);
  }

  string windowName = "First hidden layer vizualization";
  namedWindow(windowName, WINDOW_NORMAL);
  
  imshow(windowName, imagesc(image));

  waitKey(0);
}

void FFN::saveFFN(const string& filename) const{
  ofstream output(filename.c_str());
  output << nLayers() << endl;
  for (unsigned i = 0; i < nLayers(); ++i)
    output << activations_[i].rows - 1 << "\t";
  output << endl << endl;
  for (unsigned layer = 0; layer < nLayers() - 1; ++layer) {
    for (int row = 0; row < weights_[layer].rows; ++row) {
      for (int column = 0; column < weights_[layer].cols; ++column)
        output << weights_[layer].at<double>(row, column);
      output << endl;
    }
    output << endl;
  }
  output.close();
}

void FFN::loadFFN(const string& filename) {
  ifstream input(filename.c_str());
  unsigned layers_n;
  input >> layers_n;
  vector<unsigned> neurons_distr(layers_n);
  activations_.resize(layers_n);
  weights_.resize(layers_n - 1);
  weights_grads_.resize(layers_n - 1);
  local_grads_.resize(layers_n - 1);
  momentum_speed_.resize(layers_n - 1);
  for (unsigned layer = 0; layer < layers_n; ++layer) {
    input >> neurons_distr[layer];
    activations_[layer] = cv::Mat::zeros(neurons_distr[layer] + 1, 1, CV_64F);
  }
  for (unsigned layer = 0; layer < layers_n - 1; ++layer) {
    weights_[layer] = cv::Mat::zeros(neurons_distr[layer + 1], neurons_distr[layer] + 1, CV_64F);
    weights_grads_[layer] = cv::Mat::zeros(neurons_distr[layer + 1], neurons_distr[layer] + 1, CV_64F);
    momentum_speed_[layer] = cv::Mat::zeros(neurons_distr[layer + 1], neurons_distr[layer] + 1, CV_64F);
    local_grads_[layer] = cv::Mat::zeros(neurons_distr[layer + 1], 1, CV_64F);
    for (int row = 0; row < weights_[layer].rows; ++row)
    for (int column = 0; column < weights_[layer].cols; ++column)
      input >> weights_[layer].at<double>(row, column);
  }
  out_layer_type_ = SIGMOID;
  input.close();
}


