""""""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

COMMIT = "d8dced8a64d9ce305add92fa6d274fd49b569b7e"
SHA256 = "99f69118d23500cf8105c959dc292b02cb00afc667615532c6f9530d218ee9d3"

def repo():
    http_archive(
        name = "com_github_nvbench",
        urls = ["https://github.com/nvidia/nvbench/archive/{commit}.tar.gz".format(commit = COMMIT)],
        strip_prefix = "nvbench-{commit}".format(commit = COMMIT),
        sha256 = SHA256,
        build_file = "//tools/nvbench:BUILD.bazel",
        add_prefix = "nvbench",
    )
