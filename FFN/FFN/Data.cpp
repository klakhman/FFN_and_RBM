#include "Data.h"

#include <opencv2/core/core.hpp>
#include <matio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdint>

using namespace std;
using namespace cv;

Data::Data(const string& data_filename, const string& inputs_var_name, 
           const string& targets_var_name) {
  
  mat_t* matfp;
  matvar_t* matvar;
  matfp = Mat_Open(data_filename.c_str(), MAT_ACC_RDONLY);
  if (0 == matfp) { cout << "Could not open .mat file: " << data_filename <<endl; exit(1); }

  matvar = Mat_VarRead(matfp, inputs_var_name.c_str());
  inputs_ = convert(matvar);

  Mat_VarFree(matvar);
  matvar = Mat_VarRead(matfp, targets_var_name.c_str());
  targets_ = convert(matvar);
  
  Mat_VarFree(matvar);

  Mat_Close(matfp);
}

Data::Data(const string& MNIST_image_filename, const string& MNIST_target_filename){
  loadMnistImages(MNIST_image_filename);
  loadMnistLabels(MNIST_target_filename);
}

Data::Data(const string& data_filename) {
  ifstream input(data_filename.c_str());
  unsigned samples_n, input_dim, output_dim;
  input >> samples_n >> input_dim >> output_dim;
  inputs_ = Mat(input_dim, samples_n, CV_64F);
  targets_ = Mat(output_dim, samples_n, CV_64F);
  for (unsigned sample = 0; sample < samples_n; ++sample) {
    for (unsigned input_value = 0; input_value < input_dim; ++input_value)
      input >> *inputs_.ptr<double>(input_value, sample);
    for (unsigned output_value = 0; output_value < output_dim; ++output_value)
      input >> *targets_.ptr<double>(output_value, sample);
  }
  input.close();
}

Mat Data::convert(matvar_t* var) {
  const unsigned int nRows = static_cast<unsigned int> (var->dims[0]);
  const unsigned int nCols = static_cast<unsigned int> (var->dims[1]);
  Mat matrix(nRows, nCols, CV_64F);

  double* pData = static_cast<double*>(var->data);

  for (unsigned int j = 0; j < nCols; ++j)
    for (unsigned int i = 0; i < nRows; ++i)
      matrix.at<double>(i, j) = *(pData++);

  return matrix;
}

void Data::scaleUSPSdig() {
  inputs_ = (inputs_ + 1) / 2.0;
  targets_ = (targets_ + 1) / 2.0;
}

/// Загрузка изображений из базы данных MNIST
void Data::loadMnistImages(const std::string& mnistFilename){
  ifstream mnistFile(mnistFilename.c_str(), ios::binary);
  if (!mnistFile.is_open()){
    cout << "Can't open file : " << mnistFilename << endl;
    exit(11);
  }
  // В соответствии с форматом MNIST (http://yann.lecun.com/exdb/mnist/) считываем шапку
  uint32_t magicNumber, imagesNumber, rowsNumber, columnsNumber;
  mnistFile.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
  magicNumber = reverseBytewise(magicNumber);
  if (magicNumber != 2051) { cout << "Wrong MNIST images file!" << endl; exit(0); }
  mnistFile.read(reinterpret_cast<char*>(&imagesNumber), sizeof(imagesNumber));
  imagesNumber = reverseBytewise(imagesNumber);
  mnistFile.read(reinterpret_cast<char*>(&rowsNumber), sizeof(rowsNumber));
  rowsNumber = reverseBytewise(rowsNumber);
  mnistFile.read(reinterpret_cast<char*>(&columnsNumber), sizeof(columnsNumber));
  columnsNumber = reverseBytewise(columnsNumber);

  inputs_ = Mat(rowsNumber*columnsNumber, imagesNumber, CV_64F);

  // Считываем все изображения
  for (unsigned int image = 0; image < imagesNumber; ++image){
    for (unsigned int y = 0; y < rowsNumber; ++y)
      for (unsigned int x = 0; x < columnsNumber; ++x){
        uint8_t pixel;
        mnistFile.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
        inputs_.at<double>(y * columnsNumber + x, image) = pixel / 255.0;
      }
  }
}

/// Загрузка меток классов из базы данных MNIST
void Data::loadMnistLabels(const std::string& mnistFilename){
  ifstream mnistFile(mnistFilename.c_str(), ios::binary);
  if (!mnistFile.is_open()){
    cout << "Can't open file : " << mnistFilename << endl;
    exit(11);
  }
  // В соответствии с форматом MNIST (http://yann.lecun.com/exdb/mnist/) считываем шапку
  uint32_t magicNumber, labelsNumber;
  mnistFile.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
  magicNumber = reverseBytewise(magicNumber);
  if (magicNumber != 2049) { cout << "Wrong MNIST labels file!" << endl; exit(0); }
  mnistFile.read(reinterpret_cast<char*>(&labelsNumber), sizeof(labelsNumber));
  labelsNumber = reverseBytewise(labelsNumber);

  const unsigned int classesNumber = 10;
  targets_ = Mat(classesNumber, labelsNumber, CV_64F);
  for (unsigned int label = 0; label < labelsNumber; ++label){
    uint8_t labelValue;
    mnistFile.read(reinterpret_cast<char*>(&labelValue), sizeof(labelValue));
    Mat column = Mat::zeros(classesNumber, 1, CV_64F);
    column.at<double>(labelValue, 0) = 1.0;
    column.copyTo(targets_(Range(0, classesNumber), Range(label, label + 1)));
  }
}