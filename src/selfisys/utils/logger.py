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
Logger routines for the SelfiSys package.

The printing routines and colours are adapted from the Simbelmynë
comological solver (https://simbelmyne.readthedocs.io/en/latest).
"""

import sys
from typing import cast
import logging
from selfisys import DEFAULT_VERBOSE_LEVEL

# Global variables for fonts
FONT_BOLDRED = "\033[1;31m"
FONT_BOLDGREEN = "\033[1;32m"
FONT_BOLDYELLOW = "\033[1;33m"
FONT_BOLDCYAN = "\033[1;36m"
FONT_BOLDGREY = "\033[1;37m"
FONT_LIGHTPURPLE = "\033[38;5;147m"

FONT_NORMAL = "\033[00m"

# Global variables for verbosity
ERROR_VERBOSITY = 0
INFO_VERBOSITY = 1
WARNING_VERBOSITY = 2
DIAGNOSTIC_VERBOSITY = 3
DEBUG_VERBOSITY = 4
DIAGNOSTIC_LEVEL = 15
logging.addLevelName(DIAGNOSTIC_LEVEL, "DIAGNOSTIC")

G__ind__ = 0  # Global variable for logger indentation


def INDENT():
    """Indents the current level of outputs."""
    global G__ind__
    G__ind__ += 1
    return G__ind__


def UNINDENT():
    """Unindents the current level of outputs."""
    global G__ind__
    G__ind__ -= 1
    return G__ind__


def PrintLeftType(message_type, FONT_COLOR):
    """Prints the type of output to screen.

    Parameters
    ----------
    message_type (string) : type of message
    FONT_COLOR (string) : font color for this type of message

    """
    from time import localtime, strftime

    sys.stdout.write(
        "["
        + strftime("%H:%M:%S", localtime())
        + "|"
        + FONT_COLOR
        + message_type
        + FONT_NORMAL
        + "]"
    )
    sys.stdout.write("==" * G__ind__)
    sys.stdout.write("|")


def PrintInfo(message):
    """Prints an information to screen.

    Parameters
    ----------
    message (string) : message

    """
    if DEFAULT_VERBOSE_LEVEL >= INFO_VERBOSITY:
        PrintLeftType("INFO      ", FONT_BOLDCYAN)
        sys.stdout.write("{}\n".format(message))
        sys.stdout.flush()


def PrintDiagnostic(verbosity, message):
    """Prints a diagnostic to screen.

    Parameters
    ----------
    verbosity (int) : verbosity of the message
    message (string) : message

    """
    if DEFAULT_VERBOSE_LEVEL >= verbosity:
        PrintLeftType("DIAGNOSTIC", FONT_BOLDGREY)
        sys.stdout.write("{}\n".format(message))


def PrintWarning(message):
    """Prints a warning to screen.

    Parameters
    ----------
    message (string) : message

    """
    if DEFAULT_VERBOSE_LEVEL >= WARNING_VERBOSITY:
        PrintLeftType("WARNING   ", FONT_BOLDYELLOW)
        sys.stdout.write(FONT_BOLDYELLOW + message + FONT_NORMAL + "\n")


def PrintError(message):
    """Prints an error to screen.

    Parameters
    ----------
    message (string) : message

    """
    if DEFAULT_VERBOSE_LEVEL >= ERROR_VERBOSITY:
        PrintLeftType("ERROR     ", FONT_BOLDRED)
        sys.stdout.write(FONT_BOLDRED + message + FONT_NORMAL + "\n")


class CustomLoggerHandler(logging.Handler):
    """
    Custom logging handler to redirect Python logger messages to custom
    print functions, with support for verbosity levels in debug
    messages.
    """

    def emit(self, record):
        """
        Emit a log record.
        """
        try:
            log_message = self.format(record)
            log_level = record.levelno

            if log_level >= logging.ERROR:
                PrintError(log_message)
            elif log_level >= logging.WARNING:
                PrintWarning(log_message)
            elif log_level >= logging.INFO:
                PrintInfo(log_message)
            elif log_level == DIAGNOSTIC_LEVEL:
                # Retrieve verbosity level from the record
                verbosity = getattr(record, "verbosity", DIAGNOSTIC_VERBOSITY)
                PrintDiagnostic(verbosity=verbosity, message=log_message)
            elif log_level >= logging.DEBUG:
                PrintDiagnostic(verbosity=DEBUG_VERBOSITY, message=log_message)
            else:
                # Fallback for other levels
                PrintInfo(log_message)
        except Exception:
            self.handleError(record)


class CustomLogger(logging.Logger):
    """
    Custom logger class supporting custom verbosity levels in diagnostic
    messages.
    """

    def diagnostic(self, msg, *args, verbosity=DIAGNOSTIC_VERBOSITY, **kwargs) -> None:
        """
        Log a message with DIAGNOSTIC level.

        Parameters
        ----------
        msg : str
            The message to log.
        verbosity : int, optional
            The verbosity level required to log this message.
        """
        if self.isEnabledFor(DIAGNOSTIC_LEVEL):
            # Pass verbosity as part of the extra argument
            extra = kwargs.get("extra", {})
            extra["verbosity"] = verbosity
            kwargs["extra"] = extra
            self.log(DIAGNOSTIC_LEVEL, msg, *args, **kwargs)


logging.setLoggerClass(CustomLogger)


def getCustomLogger(name: str) -> CustomLogger:
    """
    Get as CustomLogger instance to use the custom printing routines.

    Parameters
    ----------
    name : str
        The name of the logger.

    Returns
    -------
    logger : logging.Logger
        The custom logger instance.
    """
    logging.setLoggerClass(CustomLogger)
    logger = cast(CustomLogger, logging.getLogger(name))  # cast for type checkers and PyLance
    logger.setLevel(logging.DEBUG)  # Set the desired base logging level

    handler = CustomLoggerHandler()
    formatter = logging.Formatter(f"{FONT_LIGHTPURPLE}(%(name)s){FONT_NORMAL} %(message)s")
    handler.setFormatter(formatter)

    # Attach the handler to the logger if not already present
    if not logger.handlers:
        logger.addHandler(handler)

    return logger
