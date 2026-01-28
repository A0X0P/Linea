// created by : A.N. Prosper
// date : January 28th 2026
// time : 16:21

#include <cstddef>

namespace Linea::IO {

struct MatrixFormat {

  std::size_t precision = 2;
  std::size_t width = 2;
  bool scientific = false;
  bool show_dimensions = false;
};

}; // namespace Linea::IO