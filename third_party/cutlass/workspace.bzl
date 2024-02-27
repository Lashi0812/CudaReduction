"""
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

COMMIT = "a75b4ac483166189a45290783cb0a18af5ff0ea5"
SHA256 = "866212a5c9e93a9a509c4b6d03484e56aa3f2d5f5db6cd2b6aaacb2b8e9404b0"

def repo():
    http_archive(
        name = "cutlass",
        sha256 = SHA256,
        strip_prefix = "cutlass-{commit}".format(commit = COMMIT),
        urls = ["https://github.com/nvidia/cutlass/archive/{commit}.tar.gz".format(commit = COMMIT)],
        build_file = "//third_party/cutlass:BUILD.bazel"
    )

