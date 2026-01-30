import warnings
from pathlib import Path

from kret_utils.filename_utils import FileSearchUtils

from .abc_lightning import ABCLM


class DeprecatedLigthningRoutes(ABCLM):
    @classmethod
    def ckpt_file_name(cls) -> Path:
        """Return the best checkpoint file by parsing val_loss from filenames.

        Expected filename format:
            best-{epoch:02d}-{val_loss:.2f}.ckpt
        Example:
            best-03-0.12.ckpt

        Searches both:
            <root>/<ModelName>/<version>/
            <root>/<ModelName>/<version>/checkpoints/
        """

        warnings.warn(
            "DeprecatedLigthningRoutes.ckpt_file_name is deprecated. Use ",
            DeprecationWarning,
            stacklevel=2,
        )

        base_folder = Path(cls._root_dir) / cls.__name__ / cls.version
        folders = [base_folder, base_folder / "checkpoints"]

        candidates: list[tuple[float, int, Path]] = []
        pattern = cls._ckpt_pattern_tuple.pattern
        matching_files = FileSearchUtils.find_matching_files(folders, pattern)

        for p in matching_files:
            name = p.name
            if "best-" not in name and "best_" not in name:
                continue
            m = pattern.search(name)
            if m is None:
                continue
            try:
                loss = float(m.group("loss"))
                epoch = int(m.group("epoch"))
            except ValueError:
                continue

            # store (-epoch) so "lowest" sorts to highest epoch when loss ties
            candidates.append((loss, -epoch, p))

        if not candidates:
            raise FileNotFoundError(
                f"No 'best' checkpoint with parsable loss found in {base_folder} or {base_folder / 'checkpoints'}"
            )

        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2]
