#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateAllTypes.h"
#endif

#ifndef THGenerateManyTypes
#define THAllLocalGenerateManyTypes
#define THGenerateManyTypes
#endif

#include "THGenerateFloatTypes.h"
#include "THGenerateIntTypes.h"

#ifdef THAllLocalGenerateManyTypes
#undef THAllLocalGenerateManyTypes
#undef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
