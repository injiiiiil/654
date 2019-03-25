#pragma once

#ifdef _WIN32
#if !defined(TORCH_BUILD_STATIC_LIBS)
#define TORCH_API CAFFE2_API
#else
#define TORCH_API
#endif
#elif defined(__GNUC__)
#if defined(torch_EXPORTS)
#define TORCH_API __attribute__((__visibility__("default")))
#else
#define TORCH_API
#endif
#else
#define TORCH_API
#endif
