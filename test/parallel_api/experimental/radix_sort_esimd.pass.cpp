// -*- C++ -*-
//===-- rasix_sort_esimd.pass.cpp -----------------------------------------===//
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

#define USE_ESIMD_SORT 1
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/ranges>

#include <sycl/sycl.hpp>
#include <vector>
#include <algorithm>
#include <random>

#include "support/utils.h"

template <typename T>
struct DescendingCompare
{
    bool
    operator()(T a, T b) const
    {
        return b < a;
    }
};

template<typename T>
void verify(const T* input, const T* ref, std::size_t size)
{
    uint32_t err_count = 0;
    for(uint32_t i = 0; i < size; ++i)
    {
        if(input[i] != ref[i])
        {
            ++err_count;
            if(err_count <= 5)
            {
                std::cout << "input[" << i << "] = " << input[i] << ", expected: " << ref[i] << std::endl;
            }
        }
    }
    if (err_count != 0)
    {
        std::cout << "error count: " << err_count << std::endl;
        std::cout << "n: " << size << std::endl;
    }
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
void generate_data(T* input, std::size_t size)
{
    std::default_random_engine gen{std::random_device{}()};
    if constexpr (std::is_integral_v<T>)
    {
        std::uniform_int_distribution<T> dist(0, size);
        std::generate(input, input + size, [&]{ return dist(gen); });
        // for(uint32_t i = 0; i < size; ++i)
        // {
        //     input[i] = i % 256;
        // }
    }
    else
    {
        std::uniform_real_distribution<T> dist(0.0, 100.0);
        std::generate(input, input + size, [&]{ return dist(gen); });
    }
}

template <typename T, typename IsAscending>
void
test_all_view(std::size_t size)
{
    namespace dpl = oneapi::dpl;
    namespace dpl_ranges = dpl::experimental::ranges;

    std::vector<T> input(size);
    generate_data(input.data(), size);
    std::vector<T> ref(input);
    if (IsAscending{})
        std::sort(std::begin(ref), std::end(ref));
    else
        std::sort(std::begin(ref), std::end(ref), DescendingCompare<T>());
    {
        sycl::buffer<T> buf(input.data(), input.size());
        dpl_ranges::all_view<T, sycl::access::mode::read_write> view(buf);
        oneapi::dpl::experimental::esimd::radix_sort(dpl::execution::dpcpp_default, view, IsAscending{});
    }
    verify(input.data(), ref.data(), size);
}

template <typename T, typename IsAscending>
void test_usm(std::size_t size)
{
    namespace dpl = oneapi::dpl;
    sycl::queue q = dpl::execution::dpcpp_default.queue();
    T* input = sycl::malloc_shared<T>(size, q);
    T* ref = sycl::malloc_host<T>(size, q);
    generate_data(ref, size);
    q.copy(ref, input, size).wait();
    if (IsAscending{})
        std::sort(ref, ref + size);
    else
        std::sort(ref, ref + size, DescendingCompare<T>());

    oneapi::dpl::experimental::esimd::radix_sort(dpl::execution::dpcpp_default, input, input + size, IsAscending{});

    T* host_input = sycl::malloc_host<T>(size, q);
    q.copy(input, host_input, size).wait();
    verify(host_input, ref, size);
}

/*
template<typename T, bool IsAscending>
void test_subrange_view(uint32_t n, sycl::queue& q)
{
    namespace dpl = oneapi::dpl;
    namespace dpl_ranges = dpl::experimental::ranges;

    T* p_in = sycl::malloc_shared<T>(n, q);
    generate_data(p_in, n);
    std::vector<T> ref(p_in, p_in + p);
    std::sort(ref);
    {
        dpl_ranges::views::subrange<T, sycl::access::mode::read_write> view(p_in, p_in + n);
        oneapi::dpl::experimental::esimd::radix_sort<decltype(dpl::execution::dpcpp_default), decltype(view),
                                                     IsAscending>(dpl::execution::dpcpp_default, view);
    }
    verify(input, ref);
}

template<typename T>
test_sycl_iterators()
{
}

*/

template <typename IsAscending>
void
test(::std::size_t size)
{
    test_all_view<uint32_t, IsAscending>(size);
    test_usm<uint32_t, IsAscending>(size);
}

int main()
{
    // std::vector<std::size_t> sizes = {16, 96, 256, 512, 2024}; // only one_wg
    // std::vector<std::size_t> sizes = {32'768, 50'000, 100'000}; // only cooperative
    // std::vector<std::size_t> sizes = {524228, 1'000'000}; // only onesweep
    std::vector<std::size_t> sizes = {16, 96, 256, 512, 2024, 32768, 50000, 524228, 1'000'000};

    for(auto size: sizes)
    {
        test<::std::true_type /* Ascending */>(size);
        test<::std::false_type /* Descending */>(size);
    }

    return TestUtils::done();
}
