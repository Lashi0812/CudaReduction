""""""

def nvbench_is_buildable():
    return select({
        "@rules_cuda//cuda:compiler_is_nvcc": [],
        "@rules_cuda//cuda:compiler_is_clang": ["@platforms//:incompatible"],
    })
