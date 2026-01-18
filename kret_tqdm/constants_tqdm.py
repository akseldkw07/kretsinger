from .typed_cls_tqdm import *


class TQDMDefaults:
    TQDM_INIT_DEF: TQDM__init__TypedDict = {
        "desc": "Update Bar",
        "leave": True,
        "position": 0,
    }


class TQDMConstants: ...
