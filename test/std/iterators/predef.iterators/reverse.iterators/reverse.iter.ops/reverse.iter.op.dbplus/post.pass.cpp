//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// constexpr reverse_iterator operator++(int);
//
//   constexpr in C++17

#include "oneapi_std_test_config.h"

#include <iostream>
#include "test_macros.h"
#include "test_iterators.h"

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(iterator)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <iterator>
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class It>
bool
test(It i, It x)
{
    s::reverse_iterator<It> r(i);
    s::reverse_iterator<It> rr = r++;
    auto ret = (r.base() == x);
    ret &= (rr.base() == i);
    return ret;
}

bool
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = true;
    {
        cl::sycl::range<1> numOfItems{1};
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                const char* s = "123";
                ret_access[0] &=
                    test(bidirectional_iterator<const char*>(s + 1), bidirectional_iterator<const char*>(s));
                ret_access[0] &=
                    test(random_access_iterator<const char*>(s + 1), random_access_iterator<const char*>(s));
                ret_access[0] &= test(s + 1, s);

#if TEST_STD_VER > 14
                {
                    constexpr const char* p = "123456789";
                    typedef s::reverse_iterator<const char*> RI;
                    constexpr RI it1 = s::make_reverse_iterator(p);
                    constexpr RI it2 = s::make_reverse_iterator(p + 1);
                    static_assert(it1 != it2, "");
                    constexpr RI it3 = s::make_reverse_iterator(p + 1)++;
                    static_assert(it1 != it3, "");
                    static_assert(it2 == it3, "");
                }
#endif
            });
        });
    }
    return ret;
}

int
main(int, char**)
{
    auto ret = kernel_test();
    if (ret)
        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;
    return 0;
}