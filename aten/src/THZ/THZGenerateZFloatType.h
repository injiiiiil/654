#ifndef THZ_GENERIC_FILE
#error "You must define THZ_GENERIC_FILE before including THZGenerateZFloatType.h"
#endif

#define THZ_NTYPE_IS_COMPLEX
#define THZ_NTYPE_IS_FPOINT
#define part float
#define Part Float

#define ntype float _Complex
#define accntype double _Complex
#define THZ_CONVERT_NTYPE_TO_ACCNTYPE(_val) (accntype)(_val)
#define THZ_CONVERT_ACCNTYPE_TO_NTYPE(_val) (ntype)(_val)
#define NType ZFloat
#define THZInf FLT_MAX
#define THZ_NTYPE_IS_ZFLOAT
#line 1 THZ_GENERIC_FILE
#include THZ_GENERIC_FILE
#undef accntype
#undef ntype
#undef NType
#undef THZInf
#undef THZ_NTYPE_IS_ZFLOAT
#undef THZ_CONVERT_NTYPE_TO_ACCNTYPE
#undef THZ_CONVERT_ACCNTYPE_TO_NTYPE

#undef part
#undef Part
#undef THZ_NTYPE_IS_COMPLEX
#undef THZ_NTYPE_IS_FPOINT

#ifndef THGenerateManyTypes
#undef THZ_GENERIC_FILE
#endif
