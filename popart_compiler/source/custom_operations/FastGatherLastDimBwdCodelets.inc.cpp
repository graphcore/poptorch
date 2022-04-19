// Copyright (c) 2022, Graphcore Ltd, All rights reserved.
#ifdef __IPU__
#include <ipu_vector_math>
#else
  #error Not supported on IPU Model
#endif
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

template<typename FT, typename IT>
struct FloatDef{
};

template<>
struct FloatDef<float, int>{
  typedef   float2    FVType;
  typedef   int2      IVType;

  static inline constexpr float2   kZeroV       = { 0.0f, 0.0f };
};

template<>
struct FloatDef<float, short>{
  typedef   float2    FVType;
  typedef   short2    IVType;
  static inline constexpr float2   kZeroV       = { 0.0f, 0.0f };
};

template<>
struct FloatDef<half, int>{
  typedef   half4     FVType;
  typedef   int2      IVType;
  static inline constexpr half4   kZeroV       = { 0.0f, 0.0f, 0.0f, 0.0f };
};

template<>
struct FloatDef<half, short>{
  typedef   half4     FVType;
  typedef   short4    IVType;
  static inline constexpr half4   kZeroV       = { 0.0f, 0.0f, 0.0f, 0.0f };
};

template<typename FT>
struct OutputDef{
};

template<>
struct OutputDef<float>{
  typedef Vector<InOut<Vector<float, ONE_PTR, 8>>>  OutputType;
};

template<>
struct OutputDef<half>{
  typedef Vector<InOut<Vector<half, ONE_PTR, 8>>>   OutputType;
};


template <class FloatType, typename IdxType> class FastGatherGradVertex : public Vertex {
public:
  FastGatherGradVertex() ;

  Vector<Input<Vector<FloatType, ONE_PTR, 8>>>  grad_out_;
  Vector<Input<Vector<IdxType, ONE_PTR, 8>>>    idx_;
  //Vector<InOut<Vector<FloatType, ONE_PTR, 8>>>  grad_in_;
  typename OutputDef<FloatType>::OutputType         grad_in_;

  const Vector<int>                             grad_out_shape_;
  const Vector<int>                             grad_in_shape_;

  template<typename FT, typename IT, typename std::enable_if<std::is_same<FT, float>::value, void>::type* = nullptr>
  static void run(Vector<Input<Vector<FT, ONE_PTR, 8>>> const&       grad_out,
                  Vector<Input<Vector<IdxType, ONE_PTR, 8>>> const&  idx,
                  typename OutputDef<FT>::OutputType&           grad_in,
                  Vector<int> const&                                 grad_out_shape,
                  Vector<int> const&                                 grad_in_shape)
  {
    int  c                      = grad_out.size();
    int  grad_out_dim_size      = grad_out_shape[grad_out_shape.size() - 1];
    int  grad_out_dim_size_half = grad_out_dim_size >> 1;
    int  grad_out_dim_size2     = grad_out_dim_size_half << 1;
    int  grad_in_dim_size       = grad_in_shape[grad_out_shape.size() - 1];
    int  grad_in_dim_size_half  = grad_in_dim_size >> 1;
    int  grad_in_dim_size2      = grad_in_dim_size_half << 1;
    for(int i = 0 ; i < c ; i ++)
    {
      typename FloatDef<FT, IT>::FVType const*  cur_grad_out_ptr2 = (typename FloatDef<FT, IT>::FVType*)(&(grad_out[i][0]));
      typename FloatDef<FT, IT>::IVType const*  cur_idx_ptr2      = (typename FloatDef<FT, IT>::IVType*)(&(idx[i][0]));
      typename FloatDef<FT, IT>::FVType*        cur_grad_in_ptr2  = (typename FloatDef<FT, IT>::FVType*)(&(grad_in[i][0]));
      
      FT const*   cur_grad_out_ptr  = (FT*)cur_grad_out_ptr2;
      IT const*   cur_idx_ptr       = (IT const*)cur_idx_ptr2;
      FT*         cur_grad_in_ptr   = (FT*)cur_grad_in_ptr2;
      int         j                 = 0;
      for(j = 0 ; j < grad_out_dim_size_half ; j ++)
      {
        typename FloatDef<FT, IT>::FVType  cur_grad_out = cur_grad_out_ptr2[j];
        typename FloatDef<FT, IT>::IVType  idx          = cur_idx_ptr2[j];
        cur_grad_in_ptr[idx[0]] += cur_grad_out[0];
        cur_grad_in_ptr[idx[1]] += cur_grad_out[1];
      }
      if(0 != (grad_out_dim_size & 1))
      {
        FT   cur_grad_out     = cur_grad_out_ptr[grad_out_dim_size2];
        IT   idx              = cur_idx_ptr[grad_out_dim_size2];
        cur_grad_in_ptr[idx] += cur_grad_out;
      }
    }
  };

  template<typename FT, typename IT, typename std::enable_if<std::is_same<FT, half>::value, void>::type* = nullptr>
  static void run(Vector<Input<Vector<FT, ONE_PTR, 8>>> const&  grad_out,
                  Vector<Input<Vector<IT, ONE_PTR, 8>>> const&  idx,
                  typename OutputDef<FT>::OutputType&      grad_in,
                  Vector<int> const&                            grad_out_shape,
                  Vector<int> const&                            grad_in_shape)
  {
    int  c                      = grad_out.size();
    int  grad_out_dim_size      = grad_out_shape[grad_out_shape.size() - 1];
    int  grad_out_dim_size_q    = grad_out_dim_size >> 2;
    int  grad_out_dim_size4     = grad_out_dim_size_q << 2;
    int  grad_in_dim_size       = grad_in_shape[grad_out_shape.size() - 1];
    int  grad_in_dim_size_q     = grad_in_dim_size >> 2;
    int  grad_in_dim_size4      = grad_out_dim_size_q << 2;
    for(int i = 0 ; i < c ; i ++)
    {
      typename FloatDef<FT, IT>::FVType const*   cur_grad_out_ptr4 = (typename FloatDef<FT, IT>::FVType*)(&(grad_out[i][0]));
      typename FloatDef<FT, IT>::IVType const*   cur_idx_ptr2      = (typename FloatDef<FT, IT>::IVType const*)(&(idx[i][0]));
      typename FloatDef<FT, IT>::FVType*         cur_grad_in_ptr4  = (typename FloatDef<FT, IT>::FVType*)(&(grad_in[i][0]));
      
      FT const*   cur_grad_out_ptr  = (FT*)cur_grad_out_ptr4;
      IT const*   cur_idx_ptr       = (IT const*)cur_idx_ptr2;
      FT*         cur_grad_in_ptr   = (FT*)cur_grad_in_ptr4;
      int          j                = 0;
      for(j = 0 ; j < grad_out_dim_size_q ; j ++)
      {
        typename FloatDef<FT, IT>::FVType   cur_grad_out = cur_grad_out_ptr4[j];
        typename FloatDef<FT, IT>::IVType   idx0         = cur_idx_ptr2[2 * j];
        typename FloatDef<FT, IT>::IVType   idx1         = cur_idx_ptr2[2 * j + 1];
        cur_grad_in_ptr[idx0[0]] += cur_grad_out[0];
        cur_grad_in_ptr[idx0[1]] += cur_grad_out[1];
        cur_grad_in_ptr[idx1[0]] += cur_grad_out[2];
        cur_grad_in_ptr[idx1[1]] += cur_grad_out[3];
      }
      for(j = grad_out_dim_size4 ; j < grad_out_dim_size ; j ++)
      {
        FT    cur_grad_out  = cur_grad_out_ptr[j];
        IT    idx           = cur_idx_ptr[j];
        cur_grad_in_ptr[idx] += cur_grad_out;
      }
    }
  }

  bool compute() {
    run<FloatType, IdxType>(grad_out_, idx_, grad_in_, grad_out_shape_, grad_in_shape_);
    return true;
  }
};

template class FastGatherGradVertex<float, int>;
template class FastGatherGradVertex<half, int>;
