//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// requires RandomAccessIterator<Iter>
//   unspecified operator[](difference_type n) const;
//
//  constexpr in C++17

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
test(It i, typename s::iterator_traits<It>::difference_type n, typename s::iterator_traits<It>::value_type x)
{
    typedef typename s::iterator_traits<It>::value_type value_type;
    const s::move_iterator<It> r(i);
    value_type rr = r[n];
    return (rr == x);
}

struct do_nothing
{
    void
    operator()(void*) const
    {
    }
};

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
                {
                    char s[] = "1234567890";
                    ret_access[0] &= test(random_access_iterator<char*>(s + 5), 4, '0');
                    ret_access[0] &= test(s + 5, 4, '0');
                }
#if TEST_STD_VER > 14
                {
                    constexpr const char* p = "123456789";
                    typedef s::move_iterator<const char*> MI;
                    constexpr MI it1 = s::make_move_iterator(p);
                    static_assert(it1[0] == '1', "");
                    static_assert(it1[5] == '6', "");
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
