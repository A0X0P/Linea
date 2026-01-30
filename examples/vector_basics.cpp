// created by : A.N. Prosper
// date : January 30th 2026
// time : 14:30

#include <Linea.hpp>
#include <iostream>

int main() {

  Linea::Vector<double> A(5);
  Linea::Vector<double> B(3, 5);
  Linea::Vector<double> C{1, 2, 3, 4, 5};
  Linea::Vector<double> D = Linea::Vector<double>::rand_fill(4, -5, 5);

  Linea::IO::display(std::cout << "Vector A: \n", A);
  Linea::IO::display(std::cout << "\nVector B: \n", B);
  Linea::IO::display(std::cout << "\nVector C: \n", C);
  Linea::IO::display(std::cout << "\nVector D: \n", D);

  return 0;
}