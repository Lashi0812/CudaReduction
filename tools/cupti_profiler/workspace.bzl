load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

COMMIT = "d134657189c121766a1256f946e1ac42d13c8f12"
SHA256 = "3c2fd329e36ebe3910dcdf87be16591e96bd0ba431be1e963828ebe154d5a057"

def repo():
    http_archive(
        name = "cupti_profiler",
        sha256 = SHA256,
        strip_prefix = "CUptiProfiler-{commit}".format(commit = COMMIT),
        urls = ["https://github.com/Lashi0812/CUptiProfiler/archive/{commit}.tar.gz".format(commit = COMMIT)],
    )
