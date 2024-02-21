load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

COMMIT = "e60cf0001fdc1500cf5ba2a6bf4ca0b46a1f725f"
SHA256 = "fa6f0aca2bf7753f91da770c601f6ee29c2d2aa251f061acd666cdca6f9dd09f"

def repo():
    http_archive(
        name = "bazel_skylib",
        sha256 = SHA256,
        strip_prefix = "bazel-skylib-{commit}".format(commit = COMMIT),
        urls = ["https://github.com/bazelbuild/bazel-skylib/archive/{commit}.tar.gz".format(commit = COMMIT)],
    )
