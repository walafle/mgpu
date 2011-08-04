// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_REDUCE_REDUCE_BLOCK_HPP
#define MGPU_BACKEND_CUDA_REDUCE_REDUCE_BLOCK_HPP

/**
 * @file reduce_blockwise.hpp
 *
 * This header provides the blockwise reduction cuda kernel interface
 */

namespace mgpu
{

template <typename T> class dev_ptr;

} // namespace mgpu

namespace mgpu
{
namespace backend_detail
{
namespace cuda
{

template <typename T, template <class> class Op>
void reduce_block(dev_ptr<T> const src, dev_ptr<T> dst,
  unsigned int const blocksize, unsigned int const blocks,
  const dev_stream & stream = default_stream);

template <typename T, template <class> class Op>
void reduce_block(dev_ptr<T> const src, T * dst,
  unsigned int const blocksize, unsigned int const blocks,
  const dev_stream & stream = default_stream);


template <typename T, template <class> class Op>
void reduce_block_p2p(dev_ptr<T> const src0, dev_ptr<T> const src1,
  dev_ptr<T> const src2, dev_ptr<T> const src3, dev_ptr<T> dst,
  unsigned int const blocksize,
  const dev_stream & stream = default_stream);

template <typename T, template <class> class Op>
void reduce_block_p2p(dev_ptr<T> const src0, dev_ptr<T> const src1,
  dev_ptr<T> const src2, dev_ptr<T> dst,
  unsigned int const blocksize,
  const dev_stream & stream = default_stream);

template <typename T, template <class> class Op>
void reduce_block_p2p(dev_ptr<T> const src0, dev_ptr<T> const src1,
  dev_ptr<T> dst, unsigned int const blocksize,
  const dev_stream & stream = default_stream);

} // namespace cuda

} // namespace backend_detail

} // namespace mgpu




#endif // MGPU_BACKEND_CUDA_REDUCE_REDUCE_BLOCK_HPP

