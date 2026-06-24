# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run static tests for TLT-pytorch project.

This script can run static tests (pylint, pydocstyle, flake8) on all modules
or only on changed files from a target branch. It supports both local and CI environments.

Usage:
    # Run tests on all modules (default behavior)
    python ci/run_static_tests.py
    
    # Run tests only on changed files from current HEAD to origin/main
    python ci/run_static_tests.py --changed-files-only
    
    # Run tests only on changed files from current HEAD to a specific branch
    python ci/run_static_tests.py --changed-files-only --target-branch origin/develop
    
    # Run tests with custom docker tag
    python ci/run_static_tests.py --tag my-tag
"""

import argparse
import os
import re
import subprocess
import sys

from utils import CI, DOCKER_ROOT, ROOT_DIR, RCFILE, TEST_MODULES
from utils import get_docker_command

STATIC_TESTS = [
    # test cases.
    "pylint --rcfile {}".format(os.path.join(DOCKER_ROOT, RCFILE)),
    "pydocstyle --ignore=D4,D107,D200,D203,D205,D210,D212,D213,D301,D400,D401",
    "flake8 --ignore=E24,W504,E501,C400,C403,C408,C409,C414,C413,C416,C417,C419"
]

# Codec-royalty guard (TAO-2183 / FF-10). Fail the static suite if a royalty-bearing
# software codec is reintroduced. Legal (2026-06-12) flagged the H.264 (libx264 / openh264),
# H.265 (libx265), and AAC software codecs; mp4v is the MPEG-4 Part 2 fourcc dropped from the
# Sparse4D writer (FF-5). Royalty-free codecs (libvpx-vp9, mjpeg) and the NVIDIA hardware path
# (h264_nvenc / h264_cuvid) are allowed.
FORBIDDEN_CODEC_PATTERNS = [
    r"libx264",
    r"libx265",
    r"openh264",
    r"mp4v",
    r"""['"]aac['"]""",
]
CODEC_SCAN_ROOTS = ["nvidia_tao_pytorch"]
# odise is third-party-derived and excluded from full-tree scans (mirrors get_changed_files /
# run_static_tests_on_all_modules); this guard lives under ci/, outside the scan roots, so its
# own pattern literals do not self-match.
CODEC_SCAN_EXCLUDE_DIRS = (os.path.join("nvidia_tao_pytorch", "cv", "odise"),)


def check_forbidden_codecs():
    """Scan the package source for reintroduced royalty-bearing codec identifiers."""
    regex = re.compile("|".join(FORBIDDEN_CODEC_PATTERNS))
    violations = []
    for scan_root in CODEC_SCAN_ROOTS:
        abs_root = os.path.join(ROOT_DIR, scan_root)
        for dirpath, _, filenames in os.walk(abs_root):
            rel_dir = os.path.relpath(dirpath, ROOT_DIR)
            if rel_dir.startswith(CODEC_SCAN_EXCLUDE_DIRS):
                continue
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                fpath = os.path.join(dirpath, filename)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as handle:
                    for lineno, line in enumerate(handle, start=1):
                        if regex.search(line):
                            violations.append(
                                "{}:{}: {}".format(
                                    os.path.relpath(fpath, ROOT_DIR), lineno, line.strip()
                                )
                            )
    if violations:
        print("ERROR: royalty-bearing codec identifier(s) found in source (TAO-2183 / FF-10):")
        print("\n".join(violations))
        print(
            "Use a royalty-free codec (libvpx-vp9 / mjpeg) or the NVIDIA hardware path "
            "(h264_nvenc / h264_cuvid). See CODEC_ROYALTY_MITIGATION_PLAN.md."
        )
    assert not violations, "Forbidden codec identifiers present in source tree."


def execute_test_command(test, target_path, docker_command_prefix, docker_image):
    """Execute a single test command on a target path.
    
    Args:
        test (str): The test command to execute
        target_path (str): The target path (file or module) to test
        docker_command_prefix (str): Docker command prefix
        docker_image (str): Docker image to use
    """
    command = f"{test} {target_path}"
    launch_command = command
    
    if not CI:
        print(f"Running test: {command}")
        launch_command = '{} -v {}:{} {} {}'.format(
            docker_command_prefix, ROOT_DIR, DOCKER_ROOT, docker_image, command
        )
    
    sys.stdout.flush()
    subprocess.check_call(launch_command, stdout=sys.stdout, stderr=sys.stdout, shell=True)


def get_changed_files(target_branch="origin/main"):
    """Get list of changed Python files from current HEAD to target branch.
    
    Args:
        target_branch (str): The target branch to compare against (default: origin/main)
        
    Returns:
        dict: Dictionary mapping modules to list of changed Python file paths 
              (excluding deleted files and files outside nvidia_tao_pytorch directory)
    """
    try:
        # Get the merge base commit between current HEAD and target branch
        merge_base_cmd = ["git", "merge-base", "HEAD", target_branch]
        merge_base = subprocess.check_output(merge_base_cmd, cwd=ROOT_DIR, text=True).strip()
        
        # Get list of changed files with status from merge base to HEAD
        # Git status codes:
        #   A = added (new file)
        #   M = modified (existing file changed)
        #   D = deleted (file removed)
        #   R = renamed (file moved/renamed)
        #   C = copied (file copied)
        # We only process A and M files, skipping D files since they can't be tested
        diff_cmd = ["git", "diff", "--name-status", merge_base, "HEAD"]
        changed_files_output = subprocess.check_output(diff_cmd, cwd=ROOT_DIR, text=True).strip()
        
        if not changed_files_output:
            return {}
        
        # Parse the output: each line is "STATUS\tFILENAME"
        # Mirror the exclusion in run_static_tests_on_all_modules: odise is
        # third-party-derived and skipped from full-tree scans.
        excluded_path_prefixes = ("nvidia_tao_pytorch/cv/odise/",)
        changed_files = []
        deleted_files = []
        for line in changed_files_output.split('\n'):
            if line.strip():
                parts = line.split('\t')
                if len(parts) == 2:
                    status, file_path = parts
                    if file_path.endswith('.py'):
                        if status in ['A', 'M']:
                            # Only include added (A) or modified (M) files under nvidia_tao_pytorch directory
                            if file_path.startswith('nvidia_tao_pytorch/'):
                                if file_path.startswith(excluded_path_prefixes):
                                    print(f"Skipping {file_path} - excluded module")
                                else:
                                    changed_files.append(file_path)
                            else:
                                print(f"Skipping {file_path} - not under nvidia_tao_pytorch directory")
                        elif status == 'D':
                            # Track deleted files for logging
                            deleted_files.append(file_path)
        
        # Log information about what files are being processed
        if deleted_files:
            newline = '\n'
            print(f"Skipping {len(deleted_files)} deleted Python files: {newline.join(deleted_files)}")
        if changed_files:
            print(f"Found {len(changed_files)} added/modified Python files under nvidia_tao_pytorch directory to test")
        else:
            print("No added or modified Python files found under nvidia_tao_pytorch directory (only deletions or files outside target directory detected)")
        
        # Group files by module for testing
        module_files = {}
        for file_path in changed_files:
            # Find which module this file belongs to
            for module in TEST_MODULES:
                if file_path.startswith(module):
                    if module not in module_files:
                        module_files[module] = []
                    module_files[module].append(file_path)
                    break
        
        return module_files
        
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not determine changed files: {e}")
        print("Falling back to running tests on all modules")
        return {}


def run_static_tests_on_changed_files(changed_files, docker_command_prefix, docker_image):
    """Run static tests only on changed files.
    
    Args:
        changed_files (dict): Dictionary mapping modules to list of changed files
        docker_command_prefix (str): Docker command prefix
        docker_image (str): Docker image to use
    """
    if not changed_files:
        print("No changed Python files found. Running tests on all modules.")
        return False
    
    print(f"Running static tests on changed files in modules: {list(changed_files.keys())}")
    
    try:
        for test in STATIC_TESTS:
            for module, files in changed_files.items():
                if not files:
                    continue
                    
                # Run test on each changed file in the module
                for file_path in files:
                    execute_test_command(test, file_path, docker_command_prefix, docker_image)
        
        return True
        
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            print(e.output)
        raise RuntimeError(f"Tests failed execution") from e
    except Exception as exc:
        raise RuntimeError(f"Tests failed with error {exc}") from exc


def run_static_tests_on_all_modules(docker_command_prefix, docker_image):
    """Run static tests on all modules.
    
    Args:
        docker_command_prefix (str): Docker command prefix
        docker_image (str): Docker image to use
    """
    try:
        for test in STATIC_TESTS:
            for module in TEST_MODULES:
                if "cv" in module:
                    submodules_to_test = [
                        os.path.join(module, item)
                        for item in os.listdir(os.path.join(ROOT_DIR, module))
                        if item not in ["odise", "__pycache__"]
                    ]
                else:
                    submodules_to_test = [module]
                
                for submodule in submodules_to_test:
                    execute_test_command(test, submodule, docker_command_prefix, docker_image)
                    
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            print(e.output)
        raise RuntimeError(f"Tests failed execution") from e
    except Exception as exc:
        raise RuntimeError(f"Tests failed with error {exc}") from exc


def parse_command_line(args=sys.argv[1:]):
    """Parse command line args for running tests if needed."""
    parser = argparse.ArgumentParser(prog="run_tests_local", description="Simple script to run tests.")
    parser.add_argument("--tag", default=None, help="Tag to the local base docker image.", type=str)
    parser.add_argument("--changed-files-only", action="store_true", 
                       help="Run tests only on changed files from current HEAD to target branch")
    parser.add_argument("--target-branch", default="origin/main", 
                       help="Target branch to compare against for changed files (default: origin/main)")
    return vars(parser.parse_args(args))


def main(cl_args=None):
    """Simple function to run local tests."""
    try:
        args = parse_command_line(cl_args)

        # Source-tree codec guard runs first (pure-Python, no docker needed).
        check_forbidden_codecs()

        manifest_file = os.path.join(ROOT_DIR, "docker/manifest.json")
        docker_command_prefix, docker_image = get_docker_command(manifest_file, args["tag"])

        # If running tests only on changed files
        if args.get("changed_files_only", False):
            changed_files = get_changed_files(args["target_branch"])
            if changed_files:
                run_static_tests_on_changed_files(changed_files, docker_command_prefix, docker_image)
                return
            else:
                print("No changed Python files found. Running tests on all modules.")

        # Run tests on all modules
        run_static_tests_on_all_modules(docker_command_prefix, docker_image)
        
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            print(e.output)
        raise RuntimeError(f"Tests failed execution") from e
    except Exception as exc:
        raise RuntimeError(f"Tests failed with error {exc}") from exc


if __name__ == "__main__":
    main(sys.argv[1:])
