import re
import unicodedata

from pathvalidate import is_valid_filename, sanitize_filename


class FilenameUtils:
    @classmethod
    def is_safe_filename_strict_re(cls, name: str) -> bool:
        # Allow alphanumeric, spaces, hyphens, underscores, periods
        if not re.match(r"^[a-zA-Z0-9._=\- ]+$", name):
            return False
        if len(name) > 255 or len(name) == 0:
            return False
        if name in {".", ".."}:
            return False
        return True

    @classmethod
    def is_safe_filename_unicode(cls, name: str) -> bool:
        # Check for null bytes
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
    def is_safe_filename_pathvalidate(cls, name: str) -> None:
        is_valid_filename(name)

    @classmethod
    def sanitize_filename_pathvalidate(cls, name: str) -> str:
        return sanitize_filename(name, replacement_text="_")
