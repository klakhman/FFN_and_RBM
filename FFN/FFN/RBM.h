#ifndef RBM_H
#define RBM_H

#include "Data.h"
#include <opencv2/core/core.hpp>
#include <string>

class RBM {
public:

  RBM(unsigned int vis_units_n, unsigned int hid_units_n);

  struct LearnParam {
    double learn_rate = 0.0;
    double momentum = 0.0;
    double weight_decay_coef = 0.0;
    unsigned int cd_k = 1;
    unsigned int mini_batch_size = 0;
  };

  void rand_weights(double range);

  void trainCD(unsigned int nIter, LearnParam learn_param, const Data& data_raw);
  void visualize_first_layer(const std::string& windowName);

  void saveRBM(const std::string& filename) const;
  void loadRBM(const std::string& filename);

  Data get_hid_representation(const Data& data_raw) const;

  friend class FFN;

private:
  Data reorder_data(const Data& data_raw);
  static cv::Mat logistic(const cv::Mat& z);

  void init_vis_biases(const cv::Mat& data_inputs);
  void init_hid_biases(const cv::Mat& data_inputs);

  cv::Mat vis_state_to_hid_probs(const cv::Mat& vis_state) const;
  cv::Mat hid_state_to_vis_probs(const cv::Mat& hid_state) const;
  cv::Mat weight_configuration_goodness_gradient(const cv::Mat& vis_state, const cv::Mat& hid_state);
  cv::Mat vis_bias_configuration_goodness_gradient(const cv::Mat& vis_state);
  cv::Mat hid_bias_configuration_goodness_gradient(const cv::Mat& hid_state);
  cv::Mat weight_decay_der();
  static cv::Mat sample_bernoulli(const cv::Mat& probs);
 
  cv::Mat imagesc(const cv::Mat& source);

  cv::Mat weights_;
  cv::Mat vis_biases_;
  cv::Mat hid_biases_;
  cv::Mat momentum_speed_;
  cv::Mat vis_bias_momentum_speed_;
  cv::Mat hid_bias_momentum_speed_;
  LearnParam learn_param_;
};

#endif // RBM_H