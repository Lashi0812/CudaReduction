""""""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

COMMIT = "344117638c8ff7e239044fd0fa7085839fc03021"
SHA256 = "8e7b955f04bc6984e4f14074d0d191474f76a6c8e849e04a9dced49bc975f2d4"

def repo():
    tf_http_archive(
        name = "com_google_benchmark",
        urls = tf_mirror_urls("https://github.com/google/benchmark/archive/{commit}.tar.gz".format(commit = COMMIT)),
        strip_prefix = "benchmark-{commit}".format(commit = COMMIT),
        sha256 = SHA256,
        build_file = "//tools/benchmark:BUILD.bazel"
    )


