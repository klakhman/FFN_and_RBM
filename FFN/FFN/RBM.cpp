#include "RBM.h"

#include "Data.h"
#include <opencv2/core/core.hpp>
#include <vector>
#include <utility>
#include <random>
#include <ctime>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

using namespace std;
using namespace cv;


RBM::RBM(unsigned int vis_units_n, unsigned int hid_units_n) {
  weights_ = cv::Mat::zeros(hid_units_n, vis_units_n, CV_64F);
  momentum_speed_ = cv::Mat::zeros(hid_units_n, vis_units_n, CV_64F);
  vis_biases_ = cv::Mat::zeros(vis_units_n, 1, CV_64F);
  vis_bias_momentum_speed_ = cv::Mat::zeros(vis_units_n, 1, CV_64F);
  hid_biases_ = cv::Mat::zeros(hid_units_n, 1, CV_64F);
  hid_bias_momentum_speed_ = cv::Mat::zeros(hid_units_n, 1, CV_64F);
}

void RBM::init_vis_biases(const Mat& data_inputs) {
  Mat average_activity;
  reduce(data_inputs, average_activity, 1, CV_REDUCE_AVG);
  divide(average_activity, 1 - average_activity, average_activity);
  log(average_activity, average_activity);
  average_activity.copyTo(vis_biases_(Range::all(), Range::all()));
}

void RBM::init_hid_biases(const Mat& data_inputs) {
  hid_biases_ = cv::Mat::ones(hid_biases_.rows, 1, CV_64F) * (-4);
  return;
}

void RBM::trainCD(unsigned int nIter, LearnParam learn_param, const Data& data_raw) {
  learn_param_ = learn_param;

  unsigned int mini_batch_n;
  const Data& data = (learn_param.mini_batch_size == 0) ? data_raw :
    reorder_data(data_raw);

  mini_batch_n = (learn_param.mini_batch_size == 0) ? 1 :
    data.get_inputs().cols / learn_param.mini_batch_size +
    ((data.get_inputs().cols % learn_param.mini_batch_size) > 0);

  const Mat& inputs = data_raw.get_inputs();


  //init_vis_biases(inputs);
  init_hid_biases(inputs);

  momentum_speed_ = 0;
  vis_bias_momentum_speed_ = 0;
  hid_bias_momentum_speed_ = 0;

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

      Mat visible_state = sample_bernoulli(batch_inputs);
      Mat hidden_state = sample_bernoulli(vis_state_to_hid_probs(visible_state));
      Mat reconstruction_vis_probs;
      Mat reconstruction_vis_state;
      Mat reconstruction_hid_probs;
      Mat reconstruction_hid_state = hidden_state.clone();
      for (unsigned int k = 0; k < learn_param_.cd_k; ++k) {
        reconstruction_vis_probs = hid_state_to_vis_probs(reconstruction_hid_state);
        reconstruction_vis_state = sample_bernoulli(reconstruction_vis_probs);
        reconstruction_hid_probs = vis_state_to_hid_probs(reconstruction_vis_state);
        reconstruction_hid_state = sample_bernoulli(reconstruction_hid_probs);
      }
      
      // Fair update
      //Mat gradient = configuration_goodness_gradient(visible_state, hidden_state) -
      //               configuration_goodness_gradient(reconstruction_vis_state, reconstruction_hid_probs);

      Mat gradient = weight_configuration_goodness_gradient(batch_inputs, hidden_state)
        - weight_configuration_goodness_gradient(reconstruction_vis_probs, reconstruction_hid_probs)
                       - learn_param_.weight_decay_coef * weight_decay_der();
      momentum_speed_ = 0.9 * momentum_speed_ + gradient;
      weights_ += learn_param_.learn_rate * momentum_speed_;

      Mat vis_bias_gradient = vis_bias_configuration_goodness_gradient(batch_inputs) -
                              vis_bias_configuration_goodness_gradient(reconstruction_vis_probs);
      vis_bias_momentum_speed_ = 0.9 * vis_bias_momentum_speed_ + vis_bias_gradient;
      vis_biases_ += learn_param_.learn_rate * vis_bias_momentum_speed_;

      Mat hid_bias_gradient = hid_bias_configuration_goodness_gradient(hidden_state) -
        hid_bias_configuration_goodness_gradient(reconstruction_hid_probs);
      hid_bias_momentum_speed_ = 0.9 * hid_bias_momentum_speed_ + hid_bias_gradient;
      hid_biases_ += learn_param_.learn_rate * hid_bias_momentum_speed_;
    }

    cout << "RBM learning iteration #" << iter << endl;
    //if ((iter + 1) % 2 == 0) 
      //visualize_first_layer("RBM at iter#" + to_string(iter));
  }
}

Mat RBM::weight_decay_der() {
  return weights_;
}

Data RBM::reorder_data(const Data& data_raw) {
  const Mat& inputs = data_raw.get_inputs();

  Data reordered_data;
  Mat& reordered_inputs = reordered_data.get_inputs_io();
  reordered_inputs = Mat(inputs.rows, inputs.cols, CV_64F);

  vector<unsigned int> indices(inputs.cols);
  unsigned int id = 0;
  for (auto& index : indices)
    index = id++;
  shuffle(begin(indices), end(indices), mt19937{ static_cast<unsigned int>(time(0)) });

  for (unsigned int index = 0; index < indices.size(); ++index) 
    inputs.col(indices[index]).copyTo(reordered_inputs.col(index));

  return reordered_data;
}

Mat RBM::sample_bernoulli(const Mat& probs) {
  Mat real_probs = Mat::zeros(probs.rows, probs.cols, CV_64F);
  randu(real_probs, Scalar::all(0), Scalar::all(1));
  Mat sample = (probs > real_probs) / 255;
  sample.convertTo(sample, CV_64F);
  return sample;
}

Mat RBM::vis_state_to_hid_probs(const Mat& vis_state) const{
  return logistic(weights_ * vis_state + repeat(hid_biases_, 1, vis_state.cols));
}

Mat RBM::hid_state_to_vis_probs(const Mat& hid_state) const{
  return logistic(weights_.t() * hid_state + repeat(vis_biases_, 1, hid_state.cols));
}

Mat RBM::weight_configuration_goodness_gradient(const Mat& vis_state, const Mat& hid_state) {
  return (hid_state * vis_state.t()) / vis_state.cols;
}

Mat RBM::vis_bias_configuration_goodness_gradient(const cv::Mat& vis_state) {
  Mat goodness;
  reduce(vis_state, goodness, 1, CV_REDUCE_AVG);
  return goodness;
}

Mat RBM::hid_bias_configuration_goodness_gradient(const cv::Mat& hid_state) {
  Mat goodness;
  reduce(hid_state, goodness, 1, CV_REDUCE_AVG);
  return goodness;
}

void RBM::rand_weights(double range) {
  randu(weights_, Scalar::all(-range), Scalar::all(range));
}

Mat RBM::logistic(const Mat& z) {
  Mat res;
  exp(-z, res);
  return 1 / (1 + res);
}

Mat RBM::imagesc(const Mat& source) {
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

void RBM::visualize_first_layer(const string& windowName) {
  unsigned int n_hid = weights_.rows;
  unsigned int n_rows = static_cast<unsigned int>(ceil(sqrt(n_hid) + 0.5));

  unsigned int mini_image_size = static_cast<unsigned int>(sqrt(weights_.cols) + 0.5);

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
    Mat mini_image = weights_.row(node);
    mini_image = mini_image.reshape(0, mini_image_size);
    Mat mini_image_pos = image(rows_range, cols_range);

    mini_image.copyTo(mini_image_pos);
    col = (++col) % n_rows;
    row += (col % n_rows == 0);
  }

  namedWindow(windowName, CV_WINDOW_KEEPRATIO);

  imshow(windowName, imagesc(image));
  //updateWindow(windowName);

  waitKey(0);
}


void RBM::saveRBM(const string& filename) const{
  ofstream output(filename.c_str());
  const unsigned int n_vis_units = weights_.cols;
  const unsigned int n_hid_units = weights_.rows;
  output << n_vis_units << "\t" << n_hid_units << endl << endl;
  for (unsigned int i = 0; i < n_vis_units; ++i)
    output << vis_biases_.at<double>(i, 0) << "\t";
  output << endl << endl;
  for (unsigned int i = 0; i < n_hid_units; ++i)
    output << hid_biases_.at<double>(i, 0) << "\t";
  output << endl << endl;
  for (unsigned int i = 0; i < n_hid_units; ++i) {
    for (unsigned int j = 0; j < n_vis_units; ++j)
      output << weights_.at<double>(i, j) << "\t";
    output << endl;
  }
  output.close();
}

void RBM::loadRBM(const string& filename) {
  ifstream input(filename.c_str());
  unsigned int n_vis_units, n_hid_units;
  input >> n_vis_units >> n_hid_units;
  vis_biases_ = Mat::zeros(n_vis_units, 1, CV_64F);
  for (unsigned int i = 0; i < n_vis_units; ++i)
    input >> vis_biases_.at<double>(i, 0);
  hid_biases_ = Mat::zeros(n_hid_units, 1, CV_64F);
  for (unsigned int i = 0; i < n_hid_units; ++i)
    input >> hid_biases_.at<double>(i, 0);
  weights_ = Mat::zeros(n_hid_units, n_vis_units, CV_64F);
  for (unsigned int i = 0; i < n_hid_units; ++i)
    for (unsigned int j = 0; j < n_vis_units; ++j)
      input >> weights_.at<double>(i, j);
  input.close();
}

Data RBM::get_hid_representation(const Data& data_raw) const{
  Data hidden_data;
  data_raw.get_targets().copyTo(hidden_data.get_targets_io());
  Mat& hidden_inputs = hidden_data.get_inputs_io();
  hidden_inputs = vis_state_to_hid_probs(data_raw.get_inputs());
  return hidden_data;
}
