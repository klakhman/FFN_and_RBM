#ifndef FFN_H
#define FFN_H

#include "Data.h"
#include <opencv2/core/core.hpp>
#include <vector>
#include <utility>
#include "RBM.h"

class FFN {
public:

  struct LearnParam {
    double learn_rate = 0.0;
    double momentum = 0.0;
    double weight_decay_coef = 0.0;
    unsigned int mini_batch_size = 0;
  };

  enum OutLayerType {
    SIGMOID,
    SOFTMAX
  };

  FFN(unsigned int nLayers, std::vector<unsigned int> neuronsDistr);

  FFN(const RBM& rbm, std::vector<unsigned int> out_layers_neurons_distr);

  FFN(const std::vector<RBM>& rbms, std::vector<unsigned int> out_layers_neurons_distr);

  unsigned int nLayers() const { return static_cast<unsigned int>(activations_.size()); }

  void set_out_layer_type(OutLayerType type) { out_layer_type_ = type; }

  double cost(const Data& data, bool raw = false);

  void rand_weights(double range);
  void rand_weights(double range, unsigned int layer_n);

  std::pair<std::vector<double>, std::vector<double> > 
    train(unsigned int nIter, LearnParam learn_param, const Data& train_data, const Data& val_data);
  
  void visualize_first_layer();

  static cv::Mat imagesc(const cv::Mat& source);

  cv::Mat calculate(const cv::Mat& inputs);

  void saveFFN(const std::string& filename) const;
  void loadFFN(const std::string& filename);

private:

  static double rand_unif_real(double A, double B); 
  static int rand_unif_int(int A, int B);
  Data reorder_train_data(const Data& train_data_raw);
  static cv::Mat logistic(const cv::Mat& z);
  void forward_pass(const cv::Mat& input);
  void calculate_out_layer();
  void backward_pass(const cv::Mat& target);
  void compute_fin_layer_grads(const cv::Mat& target);
  void compute_local_grads(unsigned int nLayer);
  void compute_weights_grads();

  void numerical_check_grads(const Data& data);

  void update_weigth();

  double cost_const(const Data& data, bool raw = false);
  double classification_error(const Data& data);


  std::vector<cv::Mat> weights_;
  std::vector<cv::Mat> activations_;
  std::vector<cv::Mat> local_grads_;
  std::vector<cv::Mat> momentum_speed_;
  std::vector<cv::Mat> weights_grads_;
  LearnParam learn_param_;
  OutLayerType out_layer_type_;
};

#endif //FFN_H
