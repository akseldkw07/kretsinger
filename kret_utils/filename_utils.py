import re
import unicodedata
from pathlib import Path

from pathvalidate import is_valid_filename, sanitize_filename


class FilenameUtils:
    @classmethod
    def is_safe_filename_basic(cls, name: Path | str) -> bool:
        try:
            Path(name).resolve()
        except Exception:
            return False

        return True

    @classmethod
    def is_safe_filename_strict_re(cls, name: Path | str) -> bool:
        # Allow alphanumeric, spaces, hyphens, underscores, periods
        name = str(name)
        if not re.match(r"^[a-zA-Z0-9._=\- ]+$", name):
            return False
        if len(name) > 255 or len(name) == 0:
            return False
        if name in {".", ".."}:
            return False
        return True

    @classmethod
    def is_safe_filename_unicode(cls, name: Path | str) -> bool:
        # Check for null bytes
        name = str(name)
        if "\x00" in name:
            return False
        # Check for path separators
        if "/" in name or "\\" in name:
            return False
        # Check for control characters
        if any(unicodedata.category(c).startswith("C") for c in name):
            return False
        # Check length (most filesystems have limits)
        if len(name) > 255:
            return False
        return True

    @classmethod
    def is_safe_filename_pathvalidate(cls, name: Path | str):
        """
        NOTE this function doesn't work well
        """
        name = str(name)
        return is_valid_filename(name)

    @classmethod
    def sanitize_filename_pathvalidate(cls, name: str) -> str:
        return sanitize_filename(name, replacement_text="_")
