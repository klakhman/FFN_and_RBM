#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "FFN.h"
#include "RBM.h"
#include "Data.h"

using namespace std;
using namespace cv;

void FFN_USPS();
void pre_trained_FFN_USPS();
void FFN_MNIST();
void pre_trained_FFN_MNIST();
void pre_trained_RBMs_FFN_MNIST();
void xor_train();


int main(int argc, char** argv) {
  xor_train();
  //FFN_MNIST();
  //pre_trained_FFN_MNIST();
  //pre_trained_RBMs_FFN_MNIST();
  return 0;
}

void xor_train() {
  Data data("C:/OpenCV/xor_dataset.txt");
  const unsigned layers_n = 3;
  const vector<unsigned> layers{ 2, 2, 1 };
  FFN net(layers_n, layers);
  net.rand_weights(0.5);

  FFN::LearnParam params;
  params.learn_rate = 0.5;
  params.momentum = 0.0;
  params.weight_decay_coef = 0;
  params.mini_batch_size = 0;

  net.train(25000, params, data, Data());
  cout << net.calculate(data.get_inputs()) << endl;
}

void FFN_USPS() {
  Data train_data("C:/OpenCV/usps_resampled.mat", 
                  "train_patterns", "train_labels");
  train_data.scaleUSPSdig();
  Data test_data("C:/OpenCV/usps_resampled.mat", 
                  "test_patterns", "test_labels");
  test_data.scaleUSPSdig();

  const unsigned int nLayers = 3;
  vector<unsigned int> layers(nLayers);
  layers[0] = 256; layers[1] = 30; layers[2] = 10;

  FFN net(nLayers, layers);
  net.rand_weights(0.01);
  net.set_out_layer_type(FFN::SOFTMAX);

  FFN::LearnParam params;
  params.learn_rate = 0.35;
  params.momentum = 0.9;
  params.weight_decay_coef = 0.0001; // 0.001
  params.mini_batch_size = 0; //100

  net.train(300, params, train_data, test_data);

  net.visualize_first_layer();

}

void pre_trained_FFN_USPS() {
  Data train_data("C:/OpenCV/usps_resampled.mat",
    "train_patterns", "train_labels");
  train_data.scaleUSPSdig();
  Data test_data("C:/OpenCV/usps_resampled.mat",
    "test_patterns", "test_labels");
  test_data.scaleUSPSdig();


  RBM rbm_net(256, 150);
  rbm_net.rand_weights(0.01);

  RBM::LearnParam rbm_params;
  rbm_params.learn_rate = 0.02;
  rbm_params.momentum = 0.9;
  rbm_params.cd_k = 5;
  rbm_params.mini_batch_size = 100;

  rbm_net.trainCD(50, rbm_params, train_data);

  rbm_net.visualize_first_layer("RBM hidden layer features");

  FFN ffn_net(rbm_net, vector<unsigned int>{10});
  ffn_net.set_out_layer_type(FFN::SOFTMAX);
  ffn_net.rand_weights(0.01, 2);

  FFN::LearnParam params;
  params.learn_rate = 0.1;
  params.momentum = 0.9;
  params.weight_decay_coef = 0; // 0.001
  params.mini_batch_size = 100;

  ffn_net.train(100, params, train_data, test_data);

  ffn_net.visualize_first_layer();
}

void FFN_MNIST() {
  Data train_data("C:/OpenCV/train-images.idx3-ubyte",
    "C:/OpenCV/train-labels.idx1-ubyte");
  
  Data test_data("C:/OpenCV/t10k-images.idx3-ubyte",
    "C:/OpenCV/t10k-labels.idx1-ubyte");

  const unsigned int nLayers = 3;
  vector<unsigned int> layers(nLayers);
  layers[0] = 28*28; layers[1] = 300; layers[2] = 10;

  FFN net(nLayers, layers);
  net.rand_weights(0.01);
  net.set_out_layer_type(FFN::SIGMOID);

  FFN::LearnParam params;
  params.learn_rate = 0.1;
  params.momentum = 0.9;
  params.weight_decay_coef = 0.0001; // 0.0001
  params.mini_batch_size = 100;

  net.train(50, params, train_data, test_data);

  net.visualize_first_layer();
}

void pre_trained_FFN_MNIST() {
  Data train_data("C:/OpenCV/train-images.idx3-ubyte",
    "C:/OpenCV/train-labels.idx1-ubyte");

  Data test_data("C:/OpenCV/t10k-images.idx3-ubyte",
    "C:/OpenCV/t10k-labels.idx1-ubyte");

  RBM rbm_net(28*28, 500);
  //rbm_net.rand_weights(0.01);

  //RBM::LearnParam rbm_params;
  //rbm_params.learn_rate = 0.1;
  //rbm_params.momentum = 0.9;
  //rbm_params.weight_decay_coef = 0.0001;
  //rbm_params.cd_k = 1;
  //rbm_params.mini_batch_size = 100;

  //rbm_net.trainCD(40, rbm_params, train_data);

  //rbm_net.saveRBM("C:/OpenCV/RBM_HU500_LR0-1_M0-9_WD1x10-4_cdK1_It20_constant_biases.txt");
  //
  //rbm_net.visualize_first_layer("RBM hidden layer features");

  rbm_net.loadRBM("C:/OpenCV/RBM_HU500_LR0-1_M0-9_WD1x10-4_cdK1_It20_constant_biases.txt");

  FFN ffn_net(rbm_net, vector<unsigned int>{100, 10});
  ffn_net.set_out_layer_type(FFN::SOFTMAX);
  ffn_net.rand_weights(0.01, 2);
  ffn_net.rand_weights(0.01, 3);

  FFN::LearnParam params;
  params.learn_rate = 0.02;
  params.momentum = 0.9;
  params.weight_decay_coef = 0.0001; // 0.0001
  params.mini_batch_size = 100;

  ffn_net.train(40, params, train_data, test_data);

  ffn_net.visualize_first_layer();
}


void pre_trained_RBMs_FFN_MNIST() {
  Data train_data("C:/OpenCV/train-images.idx3-ubyte",
    "C:/OpenCV/train-labels.idx1-ubyte");

  Data test_data("C:/OpenCV/t10k-images.idx3-ubyte",
    "C:/OpenCV/t10k-labels.idx1-ubyte");

  RBM::LearnParam rbm_params;
  rbm_params.learn_rate = 0.1;
  rbm_params.momentum = 0.9;
  rbm_params.weight_decay_coef = 0.0002;
  rbm_params.cd_k = 1;
  rbm_params.mini_batch_size = 100;

  RBM rbm_net1(28 * 28, 1000);
  rbm_net1.loadRBM("C:/OpenCV/RBM_1_4layers.txt");
  //rbm_net1.rand_weights(0.01);
  //rbm_net1.trainCD(100, rbm_params, train_data);
  //rbm_net1.saveRBM("C:/OpenCV/RBM_1_4layers.txt");
  rbm_net1.visualize_first_layer("RBM hidden layer features");
  //Data features_data = rbm_net1.get_hid_representation(train_data);

  RBM rbm_net2(1000, 500);
  rbm_net2.loadRBM("C:/OpenCV/RBM_2_4layers.txt");
  //rbm_net2.rand_weights(0.01);
  //rbm_net2.trainCD(100, rbm_params, features_data);
  //rbm_net2.saveRBM("C:/OpenCV/RBM_2_4layers.txt");
  //Data features_data_2 = rbm_net2.get_hid_representation(features_data);

  RBM rbm_net3(500, 250);
  rbm_net3.loadRBM("C:/OpenCV/RBM_3_4layers.txt");
  //rbm_net3.rand_weights(0.01);
  //rbm_net3.trainCD(100, rbm_params, features_data_2);
  //rbm_net3.saveRBM("C:/OpenCV/RBM_3_4layers.txt");
  //Data features_data_3 = rbm_net3.get_hid_representation(features_data_2);

  //RBM rbm_net4(250, 30);
  ////rbm_net4.loadRBM("C:/OpenCV/RBM_3_100it.txt");
  //rbm_net4.rand_weights(0.01);
  //rbm_net4.trainCD(100, rbm_params, features_data_3);
  //rbm_net4.saveRBM("C:/OpenCV/RBM_4_4layers.txt");

  FFN ffn_net(vector<RBM>{rbm_net1, rbm_net2, rbm_net3}, vector<unsigned int>{10});
  ffn_net.set_out_layer_type(FFN::SOFTMAX);
  ffn_net.rand_weights(0.01, 4);
  //ffn_net.rand_weights(0.01, 4);

  FFN::LearnParam params;
  params.learn_rate = 0.05;
  params.momentum = 0.9;
  params.weight_decay_coef = 0.0000; // 0.0001
  params.mini_batch_size = 100;

  ffn_net.train(50, params, train_data, test_data);

  ffn_net.visualize_first_layer();
}