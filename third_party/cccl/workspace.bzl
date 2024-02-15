"""
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

XLA_COMMIT = "36f379f29660761fe033a1306ca9dab6a88cb65c"
XLA_SHA256 = "c568de1a2fc08c9e20b161647036f2db88a28f0c651a5a800fd69873c946c37e"

def repo():
    tf_http_archive(
        name = "cccl",
        sha256 = XLA_SHA256,
        strip_prefix = "cccl-{commit}".format(commit = XLA_COMMIT),
        urls = tf_mirror_urls("https://github.com/nvidia/cccl/archive/{commit}.tar.gz".format(commit = XLA_COMMIT)),
        build_file = "//third_party/cccl:BUILD.bazel"
    )