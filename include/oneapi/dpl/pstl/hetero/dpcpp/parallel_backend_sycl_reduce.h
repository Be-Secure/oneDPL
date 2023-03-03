// -*- C++ -*-
//===-- parallel_backend_sycl_reduce.h --------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_H

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

template <typename... _Name>
class __reduce_seq_kernel;

template <typename... _Name>
class __reduce_small_kernel;

template <typename... _Name>
class __reduce_kernel;

//------------------------------------------------------------------------
// parallel_transform_reduce - async patterns
// Please see the comment for __parallel_for_submitter for optional kernel name explanation
//------------------------------------------------------------------------

// Sequential parallel_transform_reduce used for small input sizes
template <typename _Tp, typename _KernelName>
struct __parallel_transform_reduce_seq_submitter;

template <typename _Tp, typename... _Name>
struct __parallel_transform_reduce_seq_submitter<_Tp, __internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _ReduceOp, typename _TransformOp, typename _Size, typename _InitType,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
              typename... _Ranges>
    auto
    operator()(_ExecutionPolicy&& __exec, _Size __n, _ReduceOp __reduce_op, _TransformOp __transform_op,
               _InitType __init, _Ranges&&... __rngs) const
    {
        auto __transform_pattern = unseq_backend::transform_reduce_seq<_ExecutionPolicy, _ReduceOp, _TransformOp, _Tp>{
            __reduce_op, _TransformOp{__transform_op}};
        auto __init_op = unseq_backend::__init_processing<_Tp>{};

        sycl::buffer<_Tp> __res(sycl::range<1>(1));

        sycl::event __reduce_event = __exec.queue().submit([&, __n](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...); // get an access to data under SYCL buffer
            auto __res_acc = __res.template get_access<access_mode::write>(__cgh);
            __cgh.single_task<_Name...>([=] {
                _Tp __result = __transform_pattern(__n, __rngs...);
                __init_op(__init, __result, __reduce_op);
                __res_acc[0] = __result;
            });
        });

        return __future(__reduce_event, __res);
    }
};

template <typename _Tp, typename _ReduceOp, typename _TransformOp, typename _ExecutionPolicy, typename _Size,
          typename _InitType, oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
          typename... _Ranges>
auto
__parallel_transform_reduce_seq_impl(_ExecutionPolicy&& __exec, _Size __n, _ReduceOp __reduce_op,
                                     _TransformOp __transform_op, _InitType __init, _Ranges&&... __rngs)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _CustomName = typename _Policy::kernel_name;
    using _ReduceKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__reduce_seq_kernel<_CustomName>>;

    return __parallel_transform_reduce_seq_submitter<_Tp, _ReduceKernel>()(::std::forward<_ExecutionPolicy>(__exec),
                                                                           __n, __reduce_op, __transform_op, __init,
                                                                           ::std::forward<_Ranges>(__rngs)...);
}

// Parallel_transform_reduce for a single work group
template <::std::uint16_t __work_group_size, ::std::size_t __iters_per_work_item, typename _Tp, typename _KernelName>
struct __parallel_transform_reduce_small_submitter;

template <::std::uint16_t __work_group_size, ::std::size_t __iters_per_work_item, typename _Tp, typename... _Name>
struct __parallel_transform_reduce_small_submitter<__work_group_size, __iters_per_work_item, _Tp,
                                                   __internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _ReduceOp, typename _TransformOp, typename _Size, typename _InitType,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
              typename... _Ranges>
    auto
    operator()(_ExecutionPolicy&& __exec, _Size __n, _ReduceOp __reduce_op, _TransformOp __transform_op,
               _InitType __init, _Ranges&&... __rngs) const
    {
        auto __transform_pattern =
            unseq_backend::transform_reduce<_ExecutionPolicy, __iters_per_work_item, _ReduceOp, _TransformOp>{
                __reduce_op, _TransformOp{__transform_op}};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

        const _Size __n_items = __ceiling_div(__n, __iters_per_work_item); // number of work items

        sycl::buffer<_Tp> __res(sycl::range<1>(1));

        sycl::event __reduce_event = __exec.queue().submit([&, __n, __n_items](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...); // get an access to data under SYCL buffer
            auto __res_acc = __res.template get_access<access_mode::write>(__cgh);
            __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__work_group_size), __cgh);
            __cgh.parallel_for<_Name...>(
                sycl::nd_range<1>(sycl::range<1>(__work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item_id) {
                    ::std::size_t __global_idx = __item_id.get_global_id(0);
                    ::std::uint16_t __local_idx = __item_id.get_local_id(0);
                    // 1. Initialization (transform part). Fill local memory
                    __transform_pattern(__local_idx, __n, __global_idx, /*global_offset*/ 0, __temp_local, __rngs...);
                    __dpl_sycl::__group_barrier(__item_id);
                    // 2. Reduce within work group using local memory
                    _Tp __result = __reduce_pattern(__item_id, __global_idx, __n_items, __temp_local);
                    if (__local_idx == 0)
                    {
                        __reduce_pattern.apply_init(__init, __result);
                        __res_acc[0] = __result;
                    }
                });
        });

        return __future(__reduce_event, __res);
    }
}; // struct __parallel_transform_reduce_small_submitter

template <::std::uint16_t __work_group_size, ::std::size_t __iters_per_work_item, typename _Tp, typename _ReduceOp,
          typename _TransformOp, typename _ExecutionPolicy, typename _Size, typename _InitType,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_transform_reduce_small_impl(_ExecutionPolicy&& __exec, _Size __n, _ReduceOp __reduce_op,
                                       _TransformOp __transform_op, _InitType __init, _Ranges&&... __rngs)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _CustomName = typename _Policy::kernel_name;
    using _ReduceKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __reduce_small_kernel<::std::integral_constant<::std::uint16_t, __work_group_size>,
                              ::std::integral_constant<::std::size_t, __iters_per_work_item>, _CustomName>>;

    return __parallel_transform_reduce_small_submitter<__work_group_size, __iters_per_work_item, _Tp, _ReduceKernel>()(
        ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init,
        ::std::forward<_Ranges>(__rngs)...);
}

// General parallel_transform_reduce - uses a tree-based reduction
template <::std::uint16_t __work_group_size, ::std::size_t __iters_per_work_item, typename _Tp, typename _KernelName>
struct __parallel_transform_reduce_submitter;

template <::std::uint16_t __work_group_size, ::std::size_t __iters_per_work_item, typename _Tp, typename... _Name>
struct __parallel_transform_reduce_submitter<__work_group_size, __iters_per_work_item, _Tp,
                                             __internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _ReduceOp, typename _TransformOp, typename _Size, typename _InitType,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
              typename... _Ranges>
    auto
    operator()(_ExecutionPolicy&& __exec, _Size __n, _ReduceOp __reduce_op, _TransformOp __transform_op,
               _InitType __init, _Ranges&&... __rngs) const
    {
        using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;
        auto __transform_pattern1 =
            unseq_backend::transform_reduce<_ExecutionPolicy, __iters_per_work_item, _ReduceOp, _TransformOp>{
                __reduce_op, _TransformOp{__transform_op}};
        auto __transform_pattern2 =
            unseq_backend::transform_reduce<_ExecutionPolicy, __iters_per_work_item, _ReduceOp, _NoOpFunctor>{
                __reduce_op, _NoOpFunctor{}};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

        _Size __size_per_work_group =
            __iters_per_work_item * __work_group_size; // number of buffer elements processed within workgroup
        _Size __n_groups = __ceiling_div(__n, __size_per_work_group); // number of work groups
        _Size __n_items = __ceiling_div(__n, __iters_per_work_item);  // number of work items

        // Create temporary global buffers to store temporary values
        sycl::buffer<_Tp> __temp(sycl::range<1>(2 * __n_groups));
        sycl::buffer<_Tp> __res(sycl::range<1>(1));
        // __is_first == true. Reduce over each work_group
        // __is_first == false. Reduce between work groups
        bool __is_first = true;

        // For memory utilization it's better to use one big buffer instead of two small because size of the buffer is close
        // to a few MB
        _Size __offset_1 = 0;
        _Size __offset_2 = __n_groups;

        sycl::event __reduce_event;
        do
        {
            __reduce_event = __exec.queue().submit([&, __is_first, __offset_1, __offset_2, __n, __n_items,
                                                    __n_groups](sycl::handler& __cgh) {
                __cgh.depends_on(__reduce_event);

                oneapi::dpl::__ranges::__require_access(__cgh, __rngs...); // get an access to data under SYCL buffer
                auto __temp_acc = __temp.template get_access<access_mode::read_write>(__cgh);
                auto __res_acc = __res.template get_access<access_mode::write>(__cgh);
                __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__work_group_size), __cgh);
                __cgh.parallel_for<_Name...>(
                    sycl::nd_range<1>(sycl::range<1>(__n_groups * __work_group_size),
                                      sycl::range<1>(__work_group_size)),
                    [=](sycl::nd_item<1> __item_id) {
                        ::std::size_t __global_idx = __item_id.get_global_id(0);
                        ::std::uint16_t __local_idx = __item_id.get_local_id(0);
                        // 1. Initialization (transform part). Fill local memory
                        if (__is_first)
                        {
                            __transform_pattern1(__local_idx, __n, __global_idx,
                                                 /*global_offset*/ 0, __temp_local, __rngs...);
                        }
                        else
                        {
                            __transform_pattern2(__local_idx, __n, __global_idx, __offset_2, __temp_local, __temp_acc);
                        }
                        __dpl_sycl::__group_barrier(__item_id);
                        // 2. Reduce within work group using local memory
                        _Tp __result = __reduce_pattern(__item_id, __global_idx, __n_items, __temp_local);
                        if (__local_idx == 0)
                        {
                            // final reduction
                            if (__n_groups == 1)
                            {
                                __reduce_pattern.apply_init(__init, __result);
                                __res_acc[0] = __result;
                            }

                            __temp_acc[__offset_1 + __item_id.get_group(0)] = __result;
                        }
                    });
            });
            if (__is_first)
                __is_first = false;
            ::std::swap(__offset_1, __offset_2);
            __n = __n_groups;
            __n_items = __ceiling_div(__n, __iters_per_work_item);
            __n_groups = __ceiling_div(__n, __size_per_work_group);
        } while (__n > 1);

        return __future(__reduce_event, __res);
    }
}; // struct __parallel_transform_reduce_submitter

template <::std::uint16_t __work_group_size, ::std::size_t __iters_per_work_item, typename _Tp, typename _ReduceOp,
          typename _TransformOp, typename _ExecutionPolicy, typename _Size, typename _InitType,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_transform_reduce_impl(_ExecutionPolicy&& __exec, _Size __n, _ReduceOp __reduce_op,
                                 _TransformOp __transform_op, _InitType __init, _Ranges&&... __rngs)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _CustomName = typename _Policy::kernel_name;
    using _ReduceKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __reduce_kernel<::std::integral_constant<::std::uint16_t, __work_group_size>,
                        ::std::integral_constant<::std::size_t, __iters_per_work_item>, _CustomName>>;

    return __parallel_transform_reduce_submitter<__work_group_size, __iters_per_work_item, _Tp, _ReduceKernel>()(
        ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init,
        ::std::forward<_Ranges>(__rngs)...);
}

// General version of parallel_transform_reduce. Calls optimized kernels.
template <typename _Tp, typename _ReduceOp, typename _TransformOp, typename _ExecutionPolicy, typename _InitType,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_transform_reduce(_ExecutionPolicy&& __exec, _ReduceOp __reduce_op, _TransformOp __transform_op,
                            _InitType __init, _Ranges&&... __rngs)
{
    auto __n = oneapi::dpl::__ranges::__get_first_range_size(__rngs...);
    assert(__n > 0);

    // Use a single-task sequential implementation for very small arrays.
    if (__n <= 64)
    {
        return __parallel_transform_reduce_seq_impl<_Tp>(::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op,
                                                         __transform_op, __init, ::std::forward<_Ranges>(__rngs)...);
    }

    // Use a single-pass tree reduction for medium-sized arrays with the following template parameters:
    // __iters_per_work_item shows number of elements to reduce on global memory.
    // __work_group_size shows number of elements to reduce on local memory.

    // TODO: find a way to generalize getting of reliable work-group size
    ::std::size_t __work_group_size = oneapi::dpl::__internal::__max_work_group_size(__exec);
    // change __work_group_size according to local memory limit
    // Pessimistically double the memory requirement to take into account memory used by compiled kernel
    __work_group_size = oneapi::dpl::__internal::__max_local_allocation_size(::std::forward<_ExecutionPolicy>(__exec),
                                                                             sizeof(_Tp) * 2, __work_group_size);
    if (__n <= 65536 && __work_group_size >= 512)
    {
        if (__n <= 128)
        {
            return __parallel_transform_reduce_small_impl<128, 1, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                       __reduce_op, __transform_op, __init,
                                                                       ::std::forward<_Ranges>(__rngs)...);
        }
        else if (__n <= 256)
        {
            return __parallel_transform_reduce_small_impl<256, 1, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                       __reduce_op, __transform_op, __init,
                                                                       ::std::forward<_Ranges>(__rngs)...);
        }
        else if (__n <= 512)
        {
            return __parallel_transform_reduce_small_impl<256, 2, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                       __reduce_op, __transform_op, __init,
                                                                       ::std::forward<_Ranges>(__rngs)...);
        }
        else if (__n <= 1024)
        {
            return __parallel_transform_reduce_small_impl<256, 4, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                       __reduce_op, __transform_op, __init,
                                                                       ::std::forward<_Ranges>(__rngs)...);
        }
        else if (__n <= 2048)
        {
            return __parallel_transform_reduce_small_impl<256, 8, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                       __reduce_op, __transform_op, __init,
                                                                       ::std::forward<_Ranges>(__rngs)...);
        }
        else if (__n <= 4096)
        {
            return __parallel_transform_reduce_small_impl<256, 16, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                        __reduce_op, __transform_op, __init,
                                                                        ::std::forward<_Ranges>(__rngs)...);
        }
        else
        {
            return __parallel_transform_reduce_small_impl<256, 32, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n,
                                                                        __reduce_op, __transform_op, __init,
                                                                        ::std::forward<_Ranges>(__rngs)...);
        }
    }

    // Use a recursive tree reduction for large arrays.
    return __parallel_transform_reduce_impl<256, 32, _Tp>(::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op,
                                                          __transform_op, __init, ::std::forward<_Ranges>(__rngs)...);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_H
