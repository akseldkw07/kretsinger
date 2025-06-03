from __future__ import annotations

import typing as t


class LoggingLevels(t.TypedDict):
    """
    Logging levels for the Kret studies
    """

    INFO: list[str]
    WARNING: list[str]
    ERROR: list[str]
    CRITICAL: list[str]


class KretConstants:
    """
    Constants for other uses
    """

    ignore_logs: LoggingLevels = {'INFO': [], 'WARNING': [], 'ERROR': [], 'CRITICAL': []}
