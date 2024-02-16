#ifndef NVTOOLSEXT_H
#define NVTOOLSEXT_H
#include "cuda/include/nvtx3/nvToolsExt.h"
class Tracer {
public:
    Tracer(const char* name) {
        nvtxRangePushA(name);
    }
    ~Tracer() {
        nvtxRangePop();
    }
};
#define RANGE(name) Tracer uniq_name_using_macros(name);
#else
#define RANGE(name)
#endif