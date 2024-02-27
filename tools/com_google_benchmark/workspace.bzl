""""""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

COMMIT = "344117638c8ff7e239044fd0fa7085839fc03021"
SHA256 = "8e7b955f04bc6984e4f14074d0d191474f76a6c8e849e04a9dced49bc975f2d4"

def repo():
    http_archive(
        name = "com_google_benchmark",
        urls = ["https://github.com/google/benchmark/archive/{commit}.tar.gz".format(commit = COMMIT)],
        strip_prefix = "benchmark-{commit}".format(commit = COMMIT),
        sha256 = SHA256,
        build_file = "//tools/benchmark:BUILD.bazel"
    )


