from .classification_eval import ClassificationEvalUtils
from .make_dataset_wrapper import MakeSklearnDatasetsWrapper


class UTILS_sklearn(MakeSklearnDatasetsWrapper, ClassificationEvalUtils): ...
