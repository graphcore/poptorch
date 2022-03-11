R"foo(
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

template <typename FloatType, typename IdxType> class FastGatherVertex : public Vertex {
public:
  FastGatherVertex() ;

  Vector<Input<Vector<FloatType, ONE_PTR, 8>>>   data_;
  Vector<Input<Vector<IdxType, ONE_PTR, 8>>>     idx_;
  Vector<Output<Vector<FloatType, ONE_PTR, 8>>>  result_;

  const Vector<int>                              dst_shape_;

  template<typename FT, typename IT, typename std::enable_if<std::is_same<FT, float>::value, void>::type* = nullptr>
  static void run(Vector<Input<Vector<FT, ONE_PTR, 8>>> const&   data,
                  Vector<Input<Vector<IT, ONE_PTR, 8>>> const&   idx,
                  Vector<Output<Vector<FT, ONE_PTR, 8>>>&        result,
                  Vector<int> const&                             dst_shape)
  {
    int           c                = data.size();
    int           out_dim_size     = dst_shape[dst_shape.size() - 1];
    int           out_dim_size_v_r = out_dim_size >> 1;
    int           out_dim_size_v   = out_dim_size_v_r << 1;
    for(int i = 0 ; i < c ; i ++)
    {
      typename FloatDef<FT, IT>::FVType const*  cur_data_ptrv  = (typename FloatDef<FT, IT>::FVType const*)(&(data[i][0]));
      typename FloatDef<FT, IT>::IVType const*  cur_idx_ptrv   = (typename FloatDef<FT, IT>::IVType const*)(&(idx[i][0]));
      typename FloatDef<FT, IT>::FVType*        cur_out_ptrv   = (typename FloatDef<FT, IT>::FVType*)(&(result[i][0]));
      float const*   cur_data_ptr   = (float const*)cur_data_ptrv;
      int const*     cur_idx_ptr    = (int const*)cur_idx_ptrv;
      float*         cur_out_ptr    = (float*)cur_out_ptrv;
      int            j              = 0;
      for(j = 0 ; j < out_dim_size_v_r ; j ++)
      {
        typename FloatDef<FT, IT>::IVType  idx      = cur_idx_ptrv[j];
        typename FloatDef<FT, IT>::FVType  cur_val  = { cur_data_ptr[idx[0]], cur_data_ptr[idx[1]] };
        cur_out_ptrv[j]  = cur_val;
      }
      if(0 != (out_dim_size & 1))
      {
        int idx = cur_idx_ptr[out_dim_size_v];
        cur_out_ptr[out_dim_size_v] = cur_data_ptr[idx];
      }
    }
  };

  template<typename FT, typename IT, typename std::enable_if<std::is_same<FT, half>::value, void>::type* = nullptr>
  static void run(Vector<Input<Vector<FT, ONE_PTR, 8>>> const&   data,
                  Vector<Input<Vector<IT, ONE_PTR, 8>>> const&   idx,
                  Vector<Output<Vector<FT, ONE_PTR, 8>>>&        result,
                  Vector<int> const&                             dst_shape)
  {
    int           c                = data.size();
    int           out_dim_size     = dst_shape[dst_shape.size() - 1];
    int           out_dim_size_v_r = out_dim_size >> 2;
    int           out_dim_size_v   = out_dim_size_v_r << 2;
    for(int i = 0 ; i < c ; i ++)
    {
      typename FloatDef<FT, IT>::FVType const*  cur_data_ptrv = (typename FloatDef<FT, IT>::FVType const*)(&(data[i][0]));
      typename FloatDef<FT, IT>::IVType const*  cur_idx_ptrv  = (typename FloatDef<FT, IT>::IVType const*)(&(idx[i][0]));
      typename FloatDef<FT, IT>::FVType*        cur_out_ptrv  = (typename FloatDef<FT, IT>::FVType*)(&(result[i][0]));
      FT const*                                 cur_data_ptr  = (FT const*)cur_data_ptrv;
      IT const*                                 cur_idx_ptr   = (IT const*)cur_idx_ptrv;
      FT*                                       cur_out_ptr   = (FT*)cur_out_ptrv;
      int            j              = 0;
      for(j = 0 ; j < out_dim_size_v_r ; j ++)
      {
        typename FloatDef<FT, IT>::IVType   idx0      = cur_idx_ptrv[2 * j];
        typename FloatDef<FT, IT>::IVType   idx1      = cur_idx_ptrv[2 * j + 1];
        typename FloatDef<FT, IT>::FVType   cur_val   = { cur_data_ptr[idx0[0]], 
                                                          cur_data_ptr[idx0[1]],
                                                          cur_data_ptr[idx1[0]], 
                                                          cur_data_ptr[idx1[1]] };
        cur_out_ptrv[j]   = cur_val;
      }
      for(j = out_dim_size_v ; j < out_dim_size ; j ++)
      {
        IT idx = cur_idx_ptr[j];
        cur_out_ptr[j] = cur_data_ptr[idx];
      }
    }
  };

  bool compute() {
    run<FloatType, IdxType>(data_, idx_, result_, dst_shape_);
    return true;
  }
};

template class FastGatherVertex<float, int>;
template class FastGatherVertex<half, int>;
)foo"