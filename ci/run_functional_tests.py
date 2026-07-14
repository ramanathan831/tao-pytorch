# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run scripts for functional tests."""

import argparse
import os
import shlex
import subprocess
import sys

from utils import (
    CI, DOCKER_ROOT,
    ROOT_DIR
)

from utils import get_docker_command

PYTEST = "pytest -v --color=yes --testmon" # -ss


def resolve_test_path(test_path):
    """Resolve a host or relative test path to the path visible to pytest."""
    if os.path.isabs(test_path):
        if not CI and os.path.commonpath([ROOT_DIR, test_path]) == ROOT_DIR:
            return os.path.join(DOCKER_ROOT, os.path.relpath(test_path, ROOT_DIR))
        return test_path
    return os.path.join(DOCKER_ROOT, test_path)


def parse_command_line(args=sys.argv[1:]):
    """Parse command line args for running tests if needed."""
    parser = argparse.ArgumentParser(
        prog="run_tests_local",
        description="Simple script to run tests.",
        add_help=False)
    parser.add_argument(
        "--tag",
        default=None,
        help="Tag to the local base docker image.",
        type=str)
    parser.add_argument(
        "--skip_slow",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--test-path",
        action="append",
        default=[],
        help="Test path to run. Can be repeated. Defaults to the full tests directory.",
        type=str)
    args, unknown_args = parser.parse_known_args(args)
    return vars(args), unknown_args


def main(cl_args=None):
    """Simple function to run local tests."""
    try:
        args, unknown_args = parse_command_line(cl_args)
        manifest_file = os.path.join(ROOT_DIR, "docker/manifest.json")
        docker_command_prefix, docker_image = get_docker_command(
            manifest_file, args["tag"]
        )
        test_paths = args["test_path"] or [os.path.join(DOCKER_ROOT, "tests")]
        launcher_command = " ".join(
            [PYTEST]
            + [shlex.quote(resolve_test_path(test_path)) for test_path in test_paths]
            + [shlex.quote(arg) for arg in unknown_args]
        )
        if args["skip_slow"]:
            print("Skipping slow tests.")
            launcher_command += " -m \"not slow\""
        tao_core_path = "/tao-pt/tao-core"
        if not CI:
            # To build the required libraries.
            launcher_command = f"python -m pip install --no-build-isolation -e . && {launcher_command}"
            # We append tao-core to the pythonpath so imports will work
            # The functional tests are run on the base container which does not have the core wheel installed
            launcher_command = "{} -v {}:{} -w {} -e PYTHONPATH={}:{} {} bash -c {}".format(
                docker_command_prefix, ROOT_DIR,
                DOCKER_ROOT, DOCKER_ROOT, tao_core_path, DOCKER_ROOT,
                docker_image, shlex.quote(launcher_command))
        print(launcher_command)
        subprocess.check_call(launcher_command, stdout=sys.stdout, stderr=sys.stdout, shell=True)
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            print(e.output)
        raise RuntimeError(f"Tests failed execution") from e
    except Exception as exc:
        raise RuntimeError(f"Tests failed with error {exc}") from exc


if __name__=="__main__":
    main(sys.argv[1:])
