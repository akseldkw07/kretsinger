"""
Logging stuff
"""
from __future__ import annotations

import logging

from .constants import KretConstants as C

logger = logging.getLogger(__name__)


def ignore_logs():
    for package in C.ignore_logs['INFO']:
        logging.getLogger(package).setLevel(logging.CRITICAL)
    for package in C.ignore_logs['WARNING']:
        logging.getLogger(package).setLevel(logging.CRITICAL)
    for package in C.ignore_logs['ERROR']:
        logging.getLogger(package).setLevel(logging.CRITICAL)
    for package in C.ignore_logs['CRITICAL']:
        logging.getLogger(package).setLevel(logging.CRITICAL)
