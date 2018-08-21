#pragma once

#include <ATen/Allocator.h>
#include <ATen/ScalarType.h>
#include <ATen/ScalarTypeUtils.h>
#include <TH/THTypeConversion.hpp>

#include <ATen/core/intrusive_ptr.h>

#include <atomic>

struct THFinalizer {
  virtual void operator()() = 0;
  virtual ~THFinalizer() {};
};

namespace at {

struct Type;

struct AT_API StorageImpl : public c10::intrusive_ptr_target {
 public:
  StorageImpl() = delete;
  virtual ~StorageImpl() {};
  StorageImpl(
      at::ScalarType scalar_type,
      ptrdiff_t size,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable);
  StorageImpl(
      at::ScalarType scalar_type,
      ptrdiff_t size,
      at::Allocator* allocator,
      bool resizable);
  StorageImpl(StorageImpl&) = delete;
  StorageImpl(const StorageImpl&) = delete;
  // NB: Don't move ref count!
  StorageImpl(StorageImpl&& other) = default;
  StorageImpl& operator=(StorageImpl&& other) = default;

  // TODO: Rename this into th_data, and move it out of the class;
  // the real data shouldn't call th::from_type
  template <typename T>
  inline T* data() const {
    auto scalar_type_T = at::CTypeToScalarType<th::from_type<T>>::to();
    if (scalar_type_ != scalar_type_T) {
      AT_ERROR(
          "Attempt to access StorageImpl having data type ",
          at::toString(scalar_type_),
          " as data type ",
          at::toString(scalar_type_T));
    }
    return unsafe_data<T>();
  }

  template <typename T>
  inline T* unsafe_data() const {
    return static_cast<T*>(this->data_ptr_.get());
  }

  void release_resources() override {
    if (finalizer_) {
      (*finalizer_)();
    }
    finalizer_ = nullptr;
    data_ptr_.clear();
  }

  void operator=(const StorageImpl&) = delete;

  size_t elementSize() const {
    return at::elementSize(scalar_type_);
  }

  Type& type();

  // TODO: Rename to size() and size to size_
  ptrdiff_t size() const {
    return size_;
  };
  void set_size(ptrdiff_t size) {
    size_ = size;
  };
  bool resizable() const {
    return resizable_;
  };
  at::DataPtr& data_ptr() {
    return data_ptr_;
  };
  const at::DataPtr& data_ptr() const {
    return data_ptr_;
  };
  // Returns the previous data_ptr
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) {
    std::swap(data_ptr_, data_ptr);
    return std::move(data_ptr);
  };
  void* data() {
    return data_ptr_.get();
  };
  const void* data() const {
    return data_ptr_.get();
  };
  at::DeviceType device_type() const {
    return data_ptr_.device().type();
  }
  at::Allocator* allocator() {
    return allocator_;
  };
  at::ScalarType scalar_type() const {
    return scalar_type_;
  };
  const at::Allocator* allocator() const {
    return allocator_;
  };
  // You generally shouldn't use this method, but it is occasionally
  // useful if you want to override how a tensor will be reallocated,
  // after it was already allocated (and its initial allocator was
  // set)
  void set_allocator(at::Allocator* allocator) {
    allocator_ = allocator;
  }
  int getDevice() const {
    return data_ptr_.device().index();
  }
  void set_resizable(bool resizable) {
    resizable_ = resizable;
  }

  // You should only call these functions if you have a raw StorageImpl*
  // pointer; if you have intrusive_ptr<StorageImpl> this will be
  // handled automatically.
  //
  // TODO: Eliminate as many uses of these functions as humanly possible
  void _raw_incref() {
    auto ptr = c10::intrusive_ptr<StorageImpl>::reclaim(this);
    auto ptr_copy = ptr;
    ptr_copy.release();
    ptr.release();
  }
  void _raw_decref() {
    // Let it die
    c10::intrusive_ptr<StorageImpl>::reclaim(this);
    // NB: You still "have" a pointer, but it's now invalid.
    // If you want more safety, used the actual c10::intrusive_ptr class
  }
  StorageImpl* _raw_make_weak() {
    // NB: this is a strong reference
    auto ptr = c10::intrusive_ptr<StorageImpl>::reclaim(this);
    c10::weak_intrusive_ptr<StorageImpl> wptr(ptr);
    ptr.release();
    return wptr.release();
  }
  void _raw_weak_retain() {
    // NB: this is a weak reference
    auto wptr = c10::weak_intrusive_ptr<StorageImpl>::reclaim(this);
    auto wptr_copy = wptr;
    wptr_copy.release();
    wptr.release();
  }
  void _raw_weak_release() {
    // NB: this is a weak reference
    // Let it die
    c10::weak_intrusive_ptr<StorageImpl>::reclaim(this);
    // NB: You still "have" a pointer, but it's now invalid.
    // If you want more safety, used the actual c10::weak_intrusive_ptr class
  }
  StorageImpl* _raw_weak_lock() {
    auto wptr = c10::weak_intrusive_ptr<StorageImpl>::reclaim(this);
    auto ptr = wptr.lock();
    wptr.release();
    return ptr.release();
  }
  // This gives the STRONG refcount of a STRONG pointer
  uint32_t _raw_use_count() {
    auto ptr = c10::intrusive_ptr<StorageImpl>::reclaim(this);
    auto r = ptr.use_count();
    ptr.release();
    return r;
  }
  // This gives the STRONG refcount of a WEAK pointer
  uint32_t _raw_weak_use_count() {
    auto wptr = c10::weak_intrusive_ptr<StorageImpl>::reclaim(this);
    auto r = wptr.use_count();
    wptr.release();
    return r;
  }

 private:
  at::ScalarType scalar_type_;
  at::DataPtr data_ptr_;
  ptrdiff_t size_;
  bool resizable_;
  at::Allocator* allocator_;
  std::unique_ptr<THFinalizer> finalizer_;
};
} // namespace at
