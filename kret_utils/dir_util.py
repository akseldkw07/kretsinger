import typing as t


class DirUtils:
    """
    Utility class for directory operations.
    """

    @classmethod
    def dir(cls, obj: t.Any):
        """
        Print the attributes and methods of an object in a readable format.
        """
        attrs = dir(obj)
        attrs.sort(key=lambda s: s.lower())
        return attrs
