#pragma once

#include <atomic>
#include <cassert>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef __GXX_RTTI
#include <typeinfo>
#endif

#include <exception>

#include <c10/macros/Macros.h>
#include <c10/util/Backtrace.h>
#include <c10/util/C++17.h>
#include <c10/util/Exception.h>
#include <c10/util/IdWrapper.h>
#include <c10/util/Type.h>
#include <c10/util/TypeTraits.h>
#include <c10/util/TypeIndex.h>
#include <c10/util/flat_hash_map.h>

#include <c10/core/ScalarType.h>

/*
 * TypeIdentifier is a small type containing an id.
 * Types must be registered using CAFFE_KNOWN_TYPE() for them to have a type id.
 * If a type is registered, you can also create an object containing meta data
 * like constructor, destructor, stringified name, ... about the type by calling
 * TypeMeta::Make<T>. This returns a TypeMeta() object, which is basically just
 * a pointer to the type information, so it's cheap to pass around.
 */

// TODO: This file is still in the caffe2 namespace, despite living
// in the ATen directory.  This is because the macro
// CAFFE_KNOWN_TYPE defines a template specialization, which relies
// on the namespace of TypeMeta matching the namespace where the macro is
// called.  This requires us to fix all of the call-sites, which I want to do
// later.  So the namespace is not fixed at the moment.

// Make at::Half a fundamental type.
namespace c10 {
namespace guts {
template <>
struct is_fundamental<at::Half> : std::true_type {};
} // namespace guts
} // namespace c10

namespace caffe2 {

/**
 * A type id is a unique id for a given C++ type.
 * You need to register your types using CAFFE_KNOWN_TYPE(MyType) to be able to
 * use TypeIdentifier with custom types. This is for example used to store the
 * dtype of tensors.
 */
class C10_API TypeIdentifier final
    : public at::IdWrapper<TypeIdentifier, c10::util::type_index> {
public:
  friend std::ostream& operator<<(std::ostream& stream, TypeIdentifier typeId);
  friend constexpr bool operator<(TypeIdentifier lhs, TypeIdentifier rhs);

  /**
   * Returns the unique id for the given type T. The id is unique for the type T
   * in the sense that for any two different types, their ids are different; for
   * the same type T, the id remains the same over different calls of the
   * function. However, this is not guaranteed over different runs, as the id
   * is generated during run-time. Do NOT serialize the id for storage.
   */
  template <typename T>
  static C10_HOST_CONSTEXPR TypeIdentifier Get() noexcept {
    return TypeIdentifier(c10::util::get_type_index<T>());
  }

  static constexpr TypeIdentifier uninitialized() {
    return TypeIdentifier(c10::util::type_index{0});
  }

private:
  constexpr explicit TypeIdentifier(c10::util::type_index id) : IdWrapper(id) {}
};

// Allow usage in std::map / std::set
// TODO Disallow this and rather use std::unordered_map/set everywhere
inline constexpr bool operator<(TypeIdentifier lhs, TypeIdentifier rhs) {
  return lhs.underlyingId() < rhs.underlyingId();
}

inline std::ostream& operator<<(
    std::ostream& stream,
    caffe2::TypeIdentifier typeId) {
  return stream << typeId.underlyingId();
}

} // namespace caffe2

namespace at {
using DataType = caffe2::TypeIdentifier;
}

C10_DEFINE_HASH_FOR_IDWRAPPER(caffe2::TypeIdentifier)

namespace caffe2 {

namespace detail {

// This struct holds the actual type information. There will be
// one allocated per type. TypeMeta objects will then point to the struct
// instance for the type they're configured for.
struct TypeMetaData final {
  using New = void*();
  using PlacementNew = void(void*, size_t);
  using Copy = void(const void*, void*, size_t);
  using PlacementDelete = void(void*, size_t);
  using Delete = void(void*);

  constexpr TypeMetaData() noexcept
  : itemsize_(0),
    new_(nullptr),
    placementNew_(nullptr),
    copy_(nullptr),
    placementDelete_(nullptr),
    delete_(nullptr),
    id_(TypeIdentifier::uninitialized()),
    name_("nullptr (uninitialized)") {}

  constexpr TypeMetaData(
      size_t itemsize,
      New* newFn,
      PlacementNew* placementNew,
      Copy* copy,
      PlacementDelete* placementDelete,
      Delete* deleteFn,
      TypeIdentifier id,
      c10::string_view name) noexcept
  : itemsize_(itemsize),
    new_(newFn),
    placementNew_(placementNew),
    copy_(copy),
    placementDelete_(placementDelete),
    delete_(deleteFn),
    id_(id),
    name_(name) {}

  size_t itemsize_;
  New* new_;
  PlacementNew* placementNew_;
  Copy* copy_;
  PlacementDelete* placementDelete_;
  Delete* delete_;
  TypeIdentifier id_;
  c10::string_view name_;
};

// Mechanism for throwing errors which can't be prevented at compile time
// due to type erasure. E.g. somebody calling TypeMeta::copy() for
// non-copyable type. Right now just throws exception but is implemented
// in .cpp to manage dependencies
[[noreturn]] C10_API void _ThrowRuntimeTypeLogicError(const std::string& msg);

/**
 * Placement new function for the type.
 */
template <typename T>
inline void _PlacementNew(void* ptr, size_t n) {
  T* typed_ptr = static_cast<T*>(ptr);
  for (size_t i = 0; i < n; ++i) {
    new (typed_ptr + i) T;
  }
}

template <typename T>
inline void _PlacementNewNotDefault(void* /*ptr*/, size_t /*n*/) {
  _ThrowRuntimeTypeLogicError(
      "Type " + std::string(c10::util::get_fully_qualified_type_name<T>()) +
      " is not default-constructible.");
}

template <
    typename T,
    std::enable_if_t<std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeMetaData::PlacementNew* _PickPlacementNew() {
  return (c10::guts::is_fundamental<T>::value || std::is_pointer<T>::value)
      ? nullptr
      : &_PlacementNew<T>;
}

template <
    typename T,
    std::enable_if_t<!std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeMetaData::PlacementNew* _PickPlacementNew() {
  static_assert(
      !c10::guts::is_fundamental<T>::value && !std::is_pointer<T>::value,
      "this should have picked the other SFINAE case");
  return &_PlacementNewNotDefault<T>;
}

template <typename T>
inline void* _New() {
  return new T;
}

template <typename T>
inline void* _NewNotDefault() {
  _ThrowRuntimeTypeLogicError(
      "Type " + std::string(c10::util::get_fully_qualified_type_name<T>()) +
      " is not default-constructible.");
}

template <
    typename T,
    std::enable_if_t<std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeMetaData::New* _PickNew() {
  return &_New<T>;
}

template <
    typename T,
    std::enable_if_t<!std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeMetaData::New* _PickNew() {
  return &_NewNotDefault<T>;
}

/**
 * Typed copy function for classes.
 */
template <typename T>
inline void _Copy(const void* src, void* dst, size_t n) {
  const T* typed_src = static_cast<const T*>(src);
  T* typed_dst = static_cast<T*>(dst);
  for (size_t i = 0; i < n; ++i) {
    typed_dst[i] = typed_src[i];
  }
}

/**
 * A placeholder function for types that do not allow assignment.
 */
template <typename T>
inline void _CopyNotAllowed(const void* /*src*/, void* /*dst*/, size_t /*n*/) {
  _ThrowRuntimeTypeLogicError(
      "Type " + std::string(c10::util::get_fully_qualified_type_name<T>()) +
      " does not allow assignment.");
}

template <
    typename T,
    std::enable_if_t<std::is_copy_assignable<T>::value>* = nullptr>
inline constexpr TypeMetaData::Copy* _PickCopy() {
  return (c10::guts::is_fundamental<T>::value || std::is_pointer<T>::value)
      ? nullptr
      : &_Copy<T>;
}

template <
    typename T,
    std::enable_if_t<!std::is_copy_assignable<T>::value>* = nullptr>
inline constexpr TypeMetaData::Copy* _PickCopy() {
  static_assert(
      !c10::guts::is_fundamental<T>::value && !std::is_pointer<T>::value,
      "this should have picked the other SFINAE case");
  return &_CopyNotAllowed<T>;
}

/**
 * Destructor for non-fundamental types.
 */
template <typename T>
inline void _PlacementDelete(void* ptr, size_t n) {
  T* typed_ptr = static_cast<T*>(ptr);
  for (size_t i = 0; i < n; ++i) {
    typed_ptr[i].~T();
  }
}

template <typename T>
inline constexpr TypeMetaData::PlacementDelete* _PickPlacementDelete() {
  return (c10::guts::is_fundamental<T>::value || std::is_pointer<T>::value)
      ? nullptr
      : &_PlacementDelete<T>;
}

template <typename T>
inline void _Delete(void* ptr) {
  T* typed_ptr = static_cast<T*>(ptr);
  delete typed_ptr;
}

template <class T>
inline constexpr TypeMetaData::Delete* _PickDelete() noexcept {
  return &_Delete<T>;
}

class _Uninitialized final {};

} // namespace detail

//
// note: these are outside TypeMeta bc gcc seems to have trouble
// with scalarTypeItemSizes as a constexpr static member used by
// a public inline instance method
//
static constexpr uint16_t NumScalarTypes = static_cast<uint16_t>(ScalarType::NumOptions);

// item sizes for TypeMeta::itemsize() fast path
static constexpr size_t scalarTypeItemSizes[NumScalarTypes] = {
#define SCALAR_TYPE_SIZE(T, name) sizeof(T),
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SCALAR_TYPE_SIZE)
#undef SCALAR_TYPE_SIZE
    0, // Undefined
};

/**
 * TypeMeta is a thin class that allows us to store the type of a container such
 * as a blob, or the data type of a tensor, with a unique run-time id. It also
 * stores some additional data such as the item size and the name of the type
 * for run-time inspection.
 */
class C10_API TypeMeta final {
 public:
  using New = detail::TypeMetaData::New;
  using PlacementNew = detail::TypeMetaData::PlacementNew;
  using Copy = detail::TypeMetaData::Copy;
  using PlacementDelete = detail::TypeMetaData::PlacementDelete;
  using Delete = detail::TypeMetaData::Delete;

  /** Create a dummy TypeMeta object. To create a TypeMeta object for a specific
   * type, use TypeMeta::Make<T>().
   */
  TypeMeta() noexcept;

  /**
   * Copy constructor.
   */
  TypeMeta(const TypeMeta& src) noexcept = default;

  /**
   * Assignment operator.
   */
  TypeMeta& operator=(const TypeMeta& src) noexcept = default;

  TypeMeta(TypeMeta&& rhs) noexcept = default;

private:
  // TypeMeta can only be created by Make, making sure that we do not
  // create incorrectly mixed up TypeMeta objects.
  explicit TypeMeta(const uint16_t index) noexcept
  : index_(index) {
  }

 public:
  /**
   * Returns the type id.
   */
  TypeIdentifier id() const noexcept {
    return data().id_;
  }
  /**
   * true if we represent some ScalarType type
   */
  inline bool isScalarType() const noexcept {
    return index_ < NumScalarTypes;
  }
  /**
   * true if we represent ScalarType scalar_type
   */
  inline bool isScalarType(ScalarType scalar_type) const noexcept {
    return index_ == static_cast<uint16_t>(scalar_type);
  }
  /**
   * Returns the size of the item.
   */
  inline size_t itemsize() const noexcept {
    if (C10_LIKELY(isScalarType())) {
      return scalarTypeItemSizes[index_];
    }
    return data().itemsize_;
  }
  /**
   * Returns the new function pointer for individual items.
   */
  New* newFn() const noexcept {
    return data().new_;
  }
  /**
   * Returns the placement new function pointer for individual items.
   */
  PlacementNew* placementNew() const noexcept {
    return data().placementNew_;
  }
  /**
   * Returns the typed copy function pointer for individual iterms.
   */
  Copy* copy() const noexcept {
    return data().copy_;
  }
  /**
   * Returns the destructor function pointer for individual items.
   */
  PlacementDelete* placementDelete() const noexcept {
    return data().placementDelete_;
  }
  Delete* deleteFn() const noexcept {
    return data().delete_;
  }
  /**
   * Returns a printable name for the type.
   */
  c10::string_view name() const noexcept {
    return data().name_;
  }

  friend bool operator==(
      const TypeMeta lhs,
      const TypeMeta rhs) noexcept;

  template <typename T>
  bool Match() const noexcept {
    return (*this == Make<T>());
  }

  // Below are static functions that can be called by passing a specific type.

  template <class T>
  static C10_HOST_CONSTEXPR TypeIdentifier Id() noexcept {
    return TypeIdentifier::Get<T>();
  }

  template <class T>
  static c10::string_view TypeName() noexcept {
    return c10::util::get_fully_qualified_type_name<T>();
  }

  template <class T>
  static constexpr size_t ItemSize() noexcept {
    return sizeof(T);
  }

  /**
   * Returns a TypeMeta object that corresponds to the typename T.
   */
  template <typename T>
  static TypeMeta Make() {
    // The instance pointed to is declared here, but defined in a .cpp file.
    // We need to silence the compiler warning about using an undefined
    // variable template. '-Wpragmas' and '-Wunknown-warning-option' has to be
    // disabled for compilers that don't know '-Wundefined-var-template' and
    // would error at our attempt to disable it.
#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wundefined-var-template"
#endif
    return TypeMeta(_typeMetaData<T>());
#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif
  }

  /**
  * convert ScalarType enum values to TypeMeta handles
  */
  static inline caffe2::TypeMeta fromScalarType(ScalarType scalar_type) {
    const size_t index = static_cast<uint16_t>(scalar_type);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      index < NumScalarTypes,
      "Unrecognized Scalartype ", scalar_type, " (please report this error)");
    return TypeMeta(index);
  }

  /**
   * convert TypeMeta handles to ScalarType enum values
   */
  inline ScalarType toScalarType() {
    if (C10_LIKELY(isScalarType())) {
      return static_cast<ScalarType>(index_);
    }
    error_unsupported_typemeta(*this);
  }

private:
  [[noreturn]] static void error_unsupported_typemeta(caffe2::TypeMeta dtype);

  // hard limit number of registered types
  // note: constexpr provokes Windows compilation error "member may not be initialized"
  // static constexpr size_t MaxTypeIndex = UINT8_MAX;
  #define MaxTypeIndex UINT8_MAX

  static std::atomic<uint16_t> nextTypeIndex;

  static detail::TypeMetaData* typeMetaDatas();

  template <class T>
  static uint16_t addTypeMetaData() {
    const uint16_t index = nextTypeIndex++;
    TORCH_CHECK(index <= MaxTypeIndex,
      "Maximum number of CAFFE_KNOWN_TYPE declarations has been exceeded. ",
      "Please report this issue.");
    typeMetaDatas()[index] = detail::TypeMetaData{
      sizeof(T),
      detail::_PickNew<T>(),
      detail::_PickPlacementNew<T>(),
      detail::_PickCopy<T>(),
      detail::_PickPlacementDelete<T>(),
      detail::_PickDelete<T>(),
      TypeIdentifier::Get<T>(),
      c10::util::get_fully_qualified_type_name<T>()};
    return index;
  }

  // specializations return indexes into typeMetaDataInstances()
  template <class T>
  C10_API static uint16_t _typeMetaData() noexcept;

  //
  // TypeMeta just wraps this index
  //

  uint16_t index_;

  inline const detail::TypeMetaData& data() const {
    return typeMetaDatas()[index_];
  }
};

// specializations of TypeMeta::_typeMetaData for ScalarType types

#define DEFINE_SCALAR_METADATA_INSTANCE(T, name)              \
  template <>                                                 \
  constexpr uint16_t TypeMeta::_typeMetaData<T>() noexcept {  \
    return static_cast<uint16_t>(ScalarType::name);           \
  }
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_METADATA_INSTANCE)
#undef DEFINE_SCALAR_METADATA_INSTANCE

template <>
constexpr uint16_t TypeMeta::_typeMetaData<detail::_Uninitialized>() noexcept {
  return static_cast<uint16_t>(ScalarType::Undefined);
}

inline TypeMeta::TypeMeta() noexcept
  : index_(_typeMetaData<detail::_Uninitialized>()) {
}

inline bool operator==(
    const TypeMeta lhs,
    const TypeMeta rhs) noexcept {
  return (lhs.index_ == rhs.index_);
}
inline bool operator!=(
    const TypeMeta lhs,
    const TypeMeta rhs) noexcept {
  return !operator==(lhs, rhs);
}

inline std::ostream& operator<<(
    std::ostream& stream,
    caffe2::TypeMeta typeMeta) {
  return stream << typeMeta.name();
}

/**
 * Register unique id for a type so it can be used in TypeMeta context, e.g. be
 * used as a type for Blob or for Tensor elements.
 *
 * CAFFE_KNOWN_TYPE does explicit instantiation of TypeIdentifier::Get<T>
 * template function and thus needs to be put in a single translation unit (.cpp
 * file) for a given type T. Other translation units that use type T as a type
 * of the caffe2::Blob or element type of caffe2::Tensor need to depend on the
 * translation unit that contains CAFFE_KNOWN_TYPE declaration via regular
 * linkage dependencies.
 *
 * NOTE: the macro needs to be invoked in ::caffe2 namespace
 */
// Implementation note: in MSVC, we will need to prepend the C10_API
// keyword in order to get things compiled properly. in Linux, gcc seems to
// create attribute ignored error for explicit template instantiations, see
//   http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0537r0.html
//   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51930
// and as a result, we define these two macros slightly differently.
#if defined(_MSC_VER) || defined(__clang__)
#define EXPORT_IF_NOT_GCC C10_EXPORT
#else
#define EXPORT_IF_NOT_GCC
#endif

#define CAFFE_KNOWN_TYPE(T)                                           \
  template <>                                                         \
  EXPORT_IF_NOT_GCC uint16_t TypeMeta::_typeMetaData<T>() noexcept {  \
    static const uint16_t index = addTypeMetaData<T>();               \
    return index;                                                     \
  }

} // namespace caffe2
