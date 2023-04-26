# Prints accumulated Caffe2 configuration summary
function(caffe2_print_configuration_summary)
  message(STATUS "")
  message(STATUS "******** Summary ********")
  message(STATUS "General:")
  message(STATUS "  CMake version         : ${CMAKE_VERSION}")
  message(STATUS "  CMake command         : ${CMAKE_COMMAND}")
  message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
  message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
  message(STATUS "  C++ compiler id       : ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")
  message(STATUS "  Using ccache if found : ${USE_CCACHE}")
  if(USE_CCACHE)
    message(STATUS "  Found ccache          : ${CCACHE_PROGRAM}")
  endif()
  message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
  message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
  get_directory_property(tmp DIRECTORY ${PROJECT_SOURCE_DIR} COMPILE_DEFINITIONS)
  message(STATUS "  Compile definitions   : ${tmp}")
  message(STATUS "  CMAKE_PREFIX_PATH     : ${CMAKE_PREFIX_PATH}")
  message(STATUS "  CMAKE_INSTALL_PREFIX  : ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "  USE_GOLD_LINKER       : ${USE_GOLD_LINKER}")
  message(STATUS "")

  message(STATUS "  TORCH_VERSION         : ${TORCH_VERSION}")
  message(STATUS "  BUILD_CAFFE2          : ${BUILD_CAFFE2}")
  message(STATUS "  BUILD_CAFFE2_OPS      : ${BUILD_CAFFE2_OPS}")
  message(STATUS "  BUILD_STATIC_RUNTIME_BENCHMARK: ${BUILD_STATIC_RUNTIME_BENCHMARK}")
  message(STATUS "  BUILD_TENSOREXPR_BENCHMARK: ${BUILD_TENSOREXPR_BENCHMARK}")
  message(STATUS "  BUILD_NVFUSER_BENCHMARK: ${BUILD_NVFUSER_BENCHMARK}")
  message(STATUS "  BUILD_BINARY          : ${BUILD_BINARY}")
  message(STATUS "  BUILD_CUSTOM_PROTOBUF : ${BUILD_CUSTOM_PROTOBUF}")
  if(${CAFFE2_LINK_LOCAL_PROTOBUF})
    message(STATUS "    Link local protobuf : ${CAFFE2_LINK_LOCAL_PROTOBUF}")
    message(STATUS "    Use host protobuf compiler : ${USE_HOST_PROTOC}")
    if(${USE_HOST_PROTOC})
      message(STATUS "    Protobuf compiler   : ${PROTOBUF_PROTOC_EXECUTABLE}")
    endif()
  else()
    message(STATUS "    Protobuf compiler   : ${PROTOBUF_PROTOC_EXECUTABLE}")
    message(STATUS "    Protobuf includes   : ${PROTOBUF_INCLUDE_DIRS}")
    message(STATUS "    Protobuf libraries  : ${PROTOBUF_LIBRARIES}")
  endif()
  message(STATUS "  BUILD_DOCS            : ${BUILD_DOCS}")
  message(STATUS "  BUILD_PYTHON          : ${BUILD_PYTHON}")
  if(${BUILD_PYTHON})
    message(STATUS "    Python version      : ${PYTHON_VERSION_STRING}")
    message(STATUS "    Python executable   : ${PYTHON_EXECUTABLE}")
    message(STATUS "    Pythonlibs version  : ${PYTHONLIBS_VERSION_STRING}")
    message(STATUS "    Python library      : ${PYTHON_LIBRARIES}")
    message(STATUS "    Python includes     : ${PYTHON_INCLUDE_DIRS}")
    message(STATUS "    Python site-packages: ${PYTHON_SITE_PACKAGES}")
  endif()
  message(STATUS "  BUILD_SHARED_LIBS     : ${BUILD_SHARED_LIBS}")
  message(STATUS "  CAFFE2_USE_MSVC_STATIC_RUNTIME     : ${CAFFE2_USE_MSVC_STATIC_RUNTIME}")
  message(STATUS "  BUILD_TEST            : ${BUILD_TEST}")
  message(STATUS "  BUILD_JNI             : ${BUILD_JNI}")
  message(STATUS "  BUILD_MOBILE_AUTOGRAD : ${BUILD_MOBILE_AUTOGRAD}")
  message(STATUS "  BUILD_LITE_INTERPRETER: ${BUILD_LITE_INTERPRETER}")
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(STATUS "  CROSS_COMPILING_MACOSX : ${CROSS_COMPILING_MACOSX}")
  endif()
  message(STATUS "  INTERN_BUILD_MOBILE   : ${INTERN_BUILD_MOBILE}")
  message(STATUS "  TRACING_BASED         : ${TRACING_BASED}")

  message(STATUS "  USE_BLAS              : ${USE_BLAS}")
  if(${USE_BLAS})
    message(STATUS "    BLAS                : ${BLAS_INFO}")
    message(STATUS "    BLAS_HAS_SBGEMM     : ${BLAS_HAS_SBGEMM}")
  endif()
  message(STATUS "  USE_LAPACK            : ${USE_LAPACK}")
  if(${USE_LAPACK})
    message(STATUS "    LAPACK              : ${LAPACK_INFO}")
  endif()
  message(STATUS "  USE_ASAN              : ${USE_ASAN}")
  message(STATUS "  USE_TSAN              : ${USE_TSAN}")
  message(STATUS "  USE_CPP_CODE_COVERAGE : ${USE_CPP_CODE_COVERAGE}")
  message(STATUS "  USE_CUDA              : ${USE_CUDA}")
  if(${USE_CUDA})
    message(STATUS "    Split CUDA          : ${BUILD_SPLIT_CUDA}")
    message(STATUS "    CUDA static link    : ${CAFFE2_STATIC_LINK_CUDA}")
    message(STATUS "    USE_CUDNN           : ${USE_CUDNN}")
    message(STATUS "    USE_EXPERIMENTAL_CUDNN_V8_API: ${USE_EXPERIMENTAL_CUDNN_V8_API}")
    message(STATUS "    CUDA version        : ${CUDA_VERSION}")
    message(STATUS "    USE_FLASH_ATTENTION : ${USE_FLASH_ATTENTION}")
    if(${USE_CUDNN})
      message(STATUS "    cuDNN version       : ${CUDNN_VERSION}")
    endif()
    message(STATUS "    CUDA root directory : ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "    CUDA library        : ${CUDA_cuda_driver_LIBRARY}")
    message(STATUS "    cudart library      : ${CUDA_cudart_LIBRARY}")
    message(STATUS "    cublas library      : ${CUDA_cublas_LIBRARY}")
    message(STATUS "    cufft library       : ${CUDA_cufft_LIBRARY}")
    message(STATUS "    curand library      : ${CUDA_curand_LIBRARY}")
    message(STATUS "    cusparse library    : ${CUDA_cusparse_LIBRARY}")
    if(${USE_CUDNN})
      get_target_property(__tmp torch::cudnn INTERFACE_LINK_LIBRARIES)
      message(STATUS "    cuDNN library       : ${__tmp}")
    endif()
    message(STATUS "    nvrtc               : ${CUDA_nvrtc_LIBRARY}")
    message(STATUS "    CUDA include path   : ${CUDA_INCLUDE_DIRS}")
    message(STATUS "    NVCC executable     : ${CUDA_NVCC_EXECUTABLE}")
    message(STATUS "    CUDA compiler       : ${CMAKE_CUDA_COMPILER}")
    message(STATUS "    CUDA flags          : ${CMAKE_CUDA_FLAGS}")
    message(STATUS "    CUDA host compiler  : ${CMAKE_CUDA_HOST_COMPILER}")
    message(STATUS "    CUDA --device-c     : ${CUDA_SEPARABLE_COMPILATION}")
    message(STATUS "    USE_TENSORRT        : ${USE_TENSORRT}")
    if(${USE_TENSORRT})
      message(STATUS "      TensorRT runtime library: ${TENSORRT_LIBRARY}")
      message(STATUS "      TensorRT include path   : ${TENSORRT_INCLUDE_DIR}")
    endif()
  endif()
  message(STATUS "  USE_ROCM              : ${USE_ROCM}")
  if(${USE_ROCM})
    message(STATUS "    ROCM_VERSION        : ${ROCM_VERSION}")
  endif()
  message(STATUS "  BUILD_NVFUSER         : ${BUILD_NVFUSER}")
  message(STATUS "  USE_EIGEN_FOR_BLAS    : ${CAFFE2_USE_EIGEN_FOR_BLAS}")
  message(STATUS "  USE_FBGEMM            : ${USE_FBGEMM}")
  message(STATUS "    USE_FAKELOWP          : ${USE_FAKELOWP}")
  message(STATUS "  USE_KINETO            : ${USE_KINETO}")
  message(STATUS "  USE_FFMPEG            : ${USE_FFMPEG}")
  message(STATUS "  USE_GFLAGS            : ${USE_GFLAGS}")
  message(STATUS "  USE_GLOG              : ${USE_GLOG}")
  message(STATUS "  USE_LEVELDB           : ${USE_LEVELDB}")
  if(${USE_LEVELDB})
    message(STATUS "    LevelDB version     : ${LEVELDB_VERSION}")
    message(STATUS "    Snappy version      : ${Snappy_VERSION}")
  endif()
  message(STATUS "  USE_LITE_PROTO        : ${USE_LITE_PROTO}")
  message(STATUS "  USE_LMDB              : ${USE_LMDB}")
  if(${USE_LMDB})
    message(STATUS "    LMDB version        : ${LMDB_VERSION}")
  endif()
  message(STATUS "  USE_METAL             : ${USE_METAL}")
  message(STATUS "  USE_PYTORCH_METAL     : ${USE_PYTORCH_METAL}")
  message(STATUS "  USE_PYTORCH_METAL_EXPORT     : ${USE_PYTORCH_METAL_EXPORT}")
  message(STATUS "  USE_MPS               : ${USE_MPS}")
  message(STATUS "  USE_FFTW              : ${USE_FFTW}")
  message(STATUS "  USE_MKL               : ${CAFFE2_USE_MKL}")
  message(STATUS "  USE_MKLDNN            : ${USE_MKLDNN}")
  if(${USE_MKLDNN})
    message(STATUS "  USE_MKLDNN_ACL        : ${USE_MKLDNN_ACL}")
    message(STATUS "  USE_MKLDNN_CBLAS      : ${USE_MKLDNN_CBLAS}")
  endif()
  message(STATUS "  USE_UCC               : ${USE_UCC}")
  if(${USE_UCC})
    message(STATUS "    USE_SYSTEM_UCC        : ${USE_SYSTEM_UCC}")
  endif()
  message(STATUS "  USE_ITT               : ${USE_ITT}")
  message(STATUS "  USE_NCCL              : ${USE_NCCL}")
  if(${USE_NCCL})
    message(STATUS "    USE_SYSTEM_NCCL     : ${USE_SYSTEM_NCCL}")
    message(STATUS "    USE_NCCL_WITH_UCC   : ${USE_NCCL_WITH_UCC}")
  endif()
  message(STATUS "  USE_NNPACK            : ${USE_NNPACK}")
  message(STATUS "  USE_NUMPY             : ${USE_NUMPY}")
  message(STATUS "  USE_OBSERVERS         : ${USE_OBSERVERS}")
  message(STATUS "  USE_OPENCL            : ${USE_OPENCL}")
  message(STATUS "  USE_OPENCV            : ${USE_OPENCV}")
  if(${USE_OPENCV})
    message(STATUS "    OpenCV version      : ${OpenCV_VERSION}")
  endif()
  message(STATUS "  USE_OPENMP            : ${USE_OPENMP}")
  message(STATUS "  USE_TBB               : ${USE_TBB}")
  if(${USE_TBB})
    message(STATUS "    USE_SYSTEM_TBB      : ${USE_SYSTEM_TBB}")
  endif()
  message(STATUS "  USE_VULKAN            : ${USE_VULKAN}")
  if(${USE_VULKAN})
    message(STATUS "    USE_VULKAN_FP16_INFERENCE    : ${USE_VULKAN_FP16_INFERENCE}")
    message(STATUS "    USE_VULKAN_RELAXED_PRECISION : ${USE_VULKAN_RELAXED_PRECISION}")
  endif()
  message(STATUS "  USE_PROF              : ${USE_PROF}")
  message(STATUS "  USE_QNNPACK           : ${USE_QNNPACK}")
  message(STATUS "  USE_PYTORCH_QNNPACK   : ${USE_PYTORCH_QNNPACK}")
  message(STATUS "  USE_XNNPACK           : ${USE_XNNPACK}")
  message(STATUS "  USE_REDIS             : ${USE_REDIS}")
  message(STATUS "  USE_ROCKSDB           : ${USE_ROCKSDB}")
  message(STATUS "  USE_ZMQ               : ${USE_ZMQ}")
  message(STATUS "  USE_DISTRIBUTED       : ${USE_DISTRIBUTED}")
  if(${USE_DISTRIBUTED})
    message(STATUS "    USE_MPI               : ${USE_MPI}")
    message(STATUS "    USE_GLOO              : ${USE_GLOO}")
    message(STATUS "    USE_GLOO_WITH_OPENSSL : ${USE_GLOO_WITH_OPENSSL}")
    message(STATUS "    USE_TENSORPIPE        : ${USE_TENSORPIPE}")
  endif()
  if(NOT "${SELECTED_OP_LIST}" STREQUAL "")
    message(STATUS "  SELECTED_OP_LIST    : ${SELECTED_OP_LIST}")
  endif()
  message(STATUS "  Public Dependencies  : ${Caffe2_PUBLIC_DEPENDENCY_LIBS}")
  message(STATUS "  Private Dependencies : ${Caffe2_DEPENDENCY_LIBS}")
  message(STATUS "  Public CUDA Deps.    : ${Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS}")
  message(STATUS "  Private CUDA Deps.   : ${Caffe2_CUDA_DEPENDENCY_LIBS}")
  # coreml
  message(STATUS "  USE_COREML_DELEGATE     : ${USE_COREML_DELEGATE}")
  message(STATUS "  BUILD_LAZY_TS_BACKEND   : ${BUILD_LAZY_TS_BACKEND}")
  message(STATUS "  TORCH_DISABLE_GPU_ASSERTS : ${TORCH_DISABLE_GPU_ASSERTS}")
endfunction()
