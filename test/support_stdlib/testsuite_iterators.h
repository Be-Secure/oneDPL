// -*- C++ -*-
// Iterator Wrappers for the C++ library testsuite.

// This file provides the following:
//
// input_iterator_wrapper, output_iterator_wrapper
// forward_iterator_wrapper, bidirectional_iterator_wrapper and
// random_access_wrapper, which attempt to exactly perform the requirements
// of these types of iterators. These are constructed from the class
// test_container, which is given two pointers to T and an iterator type.

#ifndef _TESTSUITE_ITERATORS
#define _TESTSUITE_ITERATORS

#include <cstddef>
#include <iterator>
#include <utility>

/**
 * @brief Simple container for holding two pointers.
 *
 * Note that input_iterator_wrapper changes first to denote
 * how the valid range of == , ++, etc. change as the iterators are used.
 */
template <typename T> struct BoundsContainer {
  T *first;
  T *last;
  BoundsContainer(T *_first, T *_last) : first(_first), last(_last) {}
};

template <typename T> struct remove_cv { typedef T type; };
template <typename T> struct remove_cv<const T> { typedef T type; };
template <typename T> struct remove_cv<volatile T> { typedef T type; };
template <typename T> struct remove_cv<const volatile T> { typedef T type; };

/**
 * @brief input_iterator wrapper for pointer
 *
 * This class takes a pointer and wraps it to provide exactly
 * the requirements of a input_iterator. It should not be
 * instantiated directly, but generated from a test_container
 */
template <class T>
class input_iterator_wrapper
    : public std::iterator<std::input_iterator_tag, typename remove_cv<T>::type,
                           std::ptrdiff_t, T *, T &> {
protected:
  input_iterator_wrapper() {}

public:
  typedef BoundsContainer<T> ContainerType;
  T *ptr;
  ContainerType *SharedInfo;

  input_iterator_wrapper(T *_ptr, ContainerType *SharedInfo_in)
      : ptr(_ptr), SharedInfo(SharedInfo_in) {
    // ITERATOR_VERIFY(ptr >= SharedInfo->first && ptr <= SharedInfo->last);
  }

  input_iterator_wrapper(const input_iterator_wrapper &in)
      : ptr(in.ptr), SharedInfo(in.SharedInfo) {}

  bool operator==(const input_iterator_wrapper &in) const {
    // ITERATOR_VERIFY(SharedInfo && SharedInfo == in.SharedInfo);
    // ITERATOR_VERIFY(ptr>=SharedInfo->first && in.ptr>=SharedInfo->first);
    return ptr == in.ptr;
  }

  bool operator!=(const input_iterator_wrapper &in) const {
    return !(*this == in);
  }

  T &operator*() const {
    // ITERATOR_VERIFY(SharedInfo && ptr < SharedInfo->last);
    // ITERATOR_VERIFY(ptr >= SharedInfo->first);
    return *ptr;
  }

  T *operator->() const { return &**this; }

  input_iterator_wrapper &operator=(const input_iterator_wrapper &in) {
    ptr = in.ptr;
    SharedInfo = in.SharedInfo;
    return *this;
  }

  input_iterator_wrapper &operator++() {
    // ITERATOR_VERIFY(SharedInfo && ptr < SharedInfo->last);
    // ITERATOR_VERIFY(ptr>=SharedInfo->first);
    ptr++;
    SharedInfo->first = ptr;
    return *this;
  }

  void operator++(int) { ++*this; }

#if __cplusplus >= 201103L
  template <typename U> void operator,(const U &) const = delete;
#else
private:
  template <typename U> void operator,(const U &) const;
#endif
};

#if __cplusplus >= 201103L
template <typename T, typename U>
void operator,(const T &, const input_iterator_wrapper<U> &) = delete;
#endif

/**
 * @brief forward_iterator wrapper for pointer
 *
 * This class takes a pointer and wraps it to provide exactly
 * the requirements of a forward_iterator. It should not be
 * instantiated directly, but generated from a test_container
 */
template <class T>
struct forward_iterator_wrapper : public input_iterator_wrapper<T> {
  typedef BoundsContainer<T> ContainerType;
  typedef std::forward_iterator_tag iterator_category;
  forward_iterator_wrapper(T *_ptr, ContainerType *SharedInfo_in)
      : input_iterator_wrapper<T>(_ptr, SharedInfo_in) {}

  forward_iterator_wrapper(const forward_iterator_wrapper &in)
      : input_iterator_wrapper<T>(in) {}

  forward_iterator_wrapper() {
    this->ptr = 0;
    this->SharedInfo = 0;
  }

  T &operator*() const {
    // ITERATOR_VERIFY(this->SharedInfo && this->ptr < this->SharedInfo->last);
    return *(this->ptr);
  }

  T *operator->() const { return &**this; }

  forward_iterator_wrapper &operator++() {
    // ITERATOR_VERIFY(this->SharedInfo && this->ptr < this->SharedInfo->last);
    this->ptr++;
    return *this;
  }

  forward_iterator_wrapper operator++(int) {
    forward_iterator_wrapper<T> tmp = *this;
    ++*this;
    return tmp;
  }
};

/**
 * @brief A container-type class for holding iterator wrappers
 * test_container takes two parameters, a class T and an iterator
 * wrapper templated by T (for example forward_iterator_wrapper<T>.
 * It takes two pointers representing a range and presents them as
 * a container of iterators.
 */
template <class T, template <class TT> class ItType> struct test_container {
  typename ItType<T>::ContainerType bounds;
  test_container(T *_first, T *_last) : bounds(_first, _last) {}

#if __cplusplus >= 201103L
  template <std::size_t N>
  explicit test_container(T (&arr)[N]) : test_container(arr, arr + N) {}
#endif

  ItType<T> it(int pos) {
    // ITERATOR_VERIFY(pos >= 0 && pos <= (bounds.last - bounds.first));
    return ItType<T>(bounds.first + pos, &bounds);
  }

  ItType<T> it(T *pos) {
    // ITERATOR_VERIFY(pos >= bounds.first && pos <= bounds.last);
    return ItType<T>(pos, &bounds);
  }

  const T &val(int pos) { return (bounds.first)[pos]; }

  ItType<T> begin() { return it(bounds.first); }

  ItType<T> end() { return it(bounds.last); }
};
#endif
