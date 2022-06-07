#include "network.hpp"

inline static
double sigmoid(double x)
{
  return 1 / (1 + exp(x));
}

inline static
double sigmoid_derv(double x)
{
  return sigmoid(x) * (1 - sigmoid(x));
}

int main(int argc, char** argv)
{
  dcnn::Network net(&sigmoid, &sigmoid_derv);
  
  std::vector<double> test_data =
  {
    6.0f, 1.4535f, 0.3f, 1.0f
  };
  std::vector<double> test_target = 
  {
    0.3f, 0.7f, 0.2f
  };
  std::vector<double> test_output;

  net.propagate_forward(&test_data, &test_output);

  for(auto& val : test_output)
  {
    printf("%lf\n", val);
  }

  net.propagate_backward(&test_target);

  printf("");

  return 0;
}