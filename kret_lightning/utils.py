from __future__ import annotations


class LightningModuleAssert:
    @classmethod
    def assert_version_fmt(cls, version: str) -> None:
        if not isinstance(version, str) or not version.startswith("v_") or not version[2:].isdigit():
            raise ValueError(f"Version '{version}' is not in the correct format 'v_XXX' where XXX are digits.")
