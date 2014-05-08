#ifndef DATA_H
#define DATA_H

#include <opencv2/core/core.hpp>
#include <matio.h>
#include <string>
#include <cstdint>

class Data {
public:

  Data(const std::string& USPS_data_filename, const std::string& inputs_var_name, 
       const std::string& targets_var_name);

  Data(const std::string& MNIST_image_filename, const std::string& MNIST_target_filename);

  Data(const std::string& data_filename);

  Data() { inputs_ = cv::Mat(0, 0, CV_64F); targets_ = cv::Mat(0, 0, CV_64F); }

  const cv::Mat& get_inputs() const { return inputs_; };
  const cv::Mat& get_targets() const { return targets_; };

  cv::Mat& get_inputs_io() { return inputs_; };
  cv::Mat& get_targets_io() { return targets_; };

  void scaleUSPSdig();

private:

  /// Шаблон функции для переворота (побитово) любого значения
  template<typename I> I reverseBytewise(I value);
  /// Загрузка изображений из базы данных MNIST
  void loadMnistImages(const std::string& mnistFilename);
  /// Загрузка меток классов из базы данных MNIST
  void loadMnistLabels(const std::string& mnistFilename);

  cv::Mat convert(matvar_t* var);

  cv::Mat inputs_;
  cv::Mat targets_;
};

/// Шаблон функции для переворота (побитово) любого значения
template<typename I>
I Data::reverseBytewise(I value){
  size_t size = sizeof(value);
  I reversedValue = 0;
  for (size_t byte = 0; byte < size; ++byte){
    uint8_t byteValue = (value >> (8 * byte)) & 0xFF;
    reversedValue += static_cast<I>(byteValue) << (8 * (size - 1 - byte));
  }
  return reversedValue;
}

#endif // DATA_H