#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Copyright (C) 2024 Tristan Hoellinger
# Distributed under the GNU General Public License v3.0 (GPLv3).
# See the LICENSE file in the root directory for details.
# SPDX-License-Identifier: GPL-3.0-or-later
# ----------------------------------------------------------------------

__author__ = "Tristan Hoellinger"
__version__ = "0.1.0"
__date__ = "2024"
__license__ = "GPLv3"

"""
Tools to deal with low-level operations such as redirecting stdout from
C code.
"""

from contextlib import contextmanager
import platform
import ctypes
import io
import os, sys
import tempfile

libc = ctypes.CDLL(None)
if platform.system() == "Darwin":  # macOS
    stdout_symbol = "__stdoutp"
    stderr_symbol = "__stderrp"
else:
    stdout_symbol = "stdout"
    stderr_symbol = "stderr"
c_stdout = ctypes.c_void_p.in_dll(libc, stdout_symbol)
c_stderr = ctypes.c_void_p.in_dll(libc, stderr_symbol)


# Taken from:
# https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
@contextmanager
def stdout_redirector(stream):
    """A context manager that redirects stdout to the given stream. For
    instance, this can be used to redirect C code stdout to None (to
    avoid cluttering the log, e.g., when using tqdm).

    Args:
        stream (file-like object): The stream to which stdout should be
            redirected.
    Example:
        >>> with stdout_redirector(stream):
        >>>     print("Hello world!") # Will be printed to stream
        >>>                           # instead of stdout.
    """
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, "wb"))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode="w+b")
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)


# Adapted from:
# https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
@contextmanager
def stderr_redirector(stream):
    """A context manager that redirects stderr to the given stream.
    For instance, this can be used to redirect C code stderr to None (to
    avoid cluttering the log, e.g., when using tqdm).

    Use with caution.

    Args:
        stream (file-like object): The stream to which stdout should be
            redirected.
    """
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stderr_fd = sys.stderr.fileno()

    def _redirect_stderr(to_fd):
        """Redirect stderr to the given file descriptor."""
        # Flush the C-level buffer stderr
        libc.fflush(c_stderr)
        # Flush and close sys.stderr - also closes the file descriptor (fd)
        sys.stderr.close()
        # Make original_stderr_fd point to the same file as to_fd
        os.dup2(to_fd, original_stderr_fd)
        # Create a new sys.stderr that points to the redirected fd
        sys.stderr = io.TextIOWrapper(os.fdopen(original_stderr_fd, "wb"))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stderr_fd = os.dup(original_stderr_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode="w+b")
        _redirect_stderr(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stderr(saved_stderr_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stderr_fd)
