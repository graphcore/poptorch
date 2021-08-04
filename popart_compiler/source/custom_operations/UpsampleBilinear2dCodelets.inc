R"foo(
// Copyright (c) 2021, Graphcore Ltd, All rights reserved.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#ifdef __IPU__
#include <ipu_vector_math>
#endif

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
// #include <print.h>

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

template <typename T> class BilinearMultipleVertex : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<T, ONE_PTR>> inputs;
  poplar::Output<poplar::Vector<T>> out;
  poplar::Input<poplar::Vector<T, ONE_PTR>> w;

  bool compute() {
    unsigned int offset = 0;
    for (unsigned int i = 0; i < out.size(); ++i) {
      out[i] = inputs[offset] * w[0] + inputs[offset + 1] * w[1] +
               inputs[offset + 2] * w[2] + inputs[offset + 3] * w[3];
      offset += 4;
    }
    return true;
  }
};

template class BilinearMultipleVertex<float>;
template class BilinearMultipleVertex<half>;

template <typename T> class BilinearGradVertex : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<T>> input;
  poplar::Input<poplar::Vector<T>> w;
  poplar::Output<poplar::Vector<T>> out;

  bool compute() {
    unsigned int offset = 0;
    for (unsigned int i = 0; i < out.size(); ++i) { // b x c
      float res = 0.0f;
      for (unsigned int j = 0; j < w.size(); ++j) {
        res += float(input[offset + j] * w[j]);
      }
      out[i] = res;
      offset += w.size();
    }
    return true;
  }
};

template class BilinearGradVertex<float>;
template class BilinearGradVertex<half>;

template <typename T> class BilinearGradMultipleVertex : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<T>> input;
  poplar::Input<poplar::Vector<T>> w;
  poplar::Input<poplar::Vector<unsigned int>> limits;
  poplar::Output<poplar::Vector<T>> out;

  bool compute() {
    unsigned int offset = 0;
    const size_t block_size = out.size() / limits.size();
    for (unsigned int i = 0; i < block_size; ++i) { // b x c
      unsigned int w_offset = 0;
      unsigned int pixel = 0;
      for (unsigned int limit : limits) {
        float res = 0.0f;
        for (unsigned int j = 0; j < limit; ++j) {
          res += float(input[offset + j] * w[w_offset + j]);
        }
        out[pixel * block_size + i] = res;
        offset += limit;
        w_offset += limit;
        ++pixel;
      }
    }
    return true;
  }
};

template class BilinearGradMultipleVertex<float>;
template class BilinearGradMultipleVertex<half>;
)foo"
