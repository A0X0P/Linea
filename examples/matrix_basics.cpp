// created by : A.N. Prosper
// date : January 30th 2026
// time : 14:30

#include <Linea.hpp>
#include <iostream>

int main() {

  Linea::Matrix<double> A(5, 5);    // zero filled
  Linea::Matrix<double> B(3, 5, 5); // Initialize with value 5
  Linea::Matrix<double> C{
      {1, 2, 3, 4, 5}, {0, 4, 6, 7, 8}, {0, 5, 9, 10, 11}}; // Initializer list
  Linea::Matrix<double> D =
      Linea::Matrix<double>::rand_fill(4, 8, -5, 5);            // random fill
  Linea::Matrix<double> E = Linea::Matrix<double>::Zeros(5, 4); // zeros
  Linea::Matrix<double> F = Linea::Matrix<double>::Ones(6, 8);  // ones
  Linea::Matrix<double> G = Linea::Matrix<double>::Identity(6); // identity

  Linea::IO::display(std::cout << "Matrix A: \n", A);
  Linea::IO::display(std::cout << "\nMatrix B: \n", B);
  Linea::IO::display(std::cout << "\nMatrix C: \n", C);
  Linea::IO::display(std::cout << "\nMatrix D: \n", D);
  Linea::IO::display(std::cout << "\nMatrix E: \n", E);
  Linea::IO::display(std::cout << "\nMatrix F: \n", F);
  Linea::IO::display(std::cout << "\nMatrix G: \n", G);

  return 0;
}