from typing import Dict

import torchvision

from data.tree_datasets import (
    CocoNotatedSyntaxTreeDataset,
    CocoSyntaxTreeDataset,
)


class Singleton(type):
    _instances: Dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class S:
    """Collection singleton dataset classes"""

    class TrainCaptions(torchvision.datasets.CocoCaptions, metaclass=Singleton):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class ValCaptions(torchvision.datasets.CocoCaptions, metaclass=Singleton):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class AbsurdCaptions(torchvision.datasets.CocoCaptions, metaclass=Singleton):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class SyntaxTrainCaptions(CocoSyntaxTreeDataset, metaclass=Singleton):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class SyntaxValCaptions(CocoSyntaxTreeDataset, metaclass=Singleton):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class SyntaxAbsurdCaptions(CocoSyntaxTreeDataset, metaclass=Singleton):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class NotatedSyntaxTrainCaptions(CocoNotatedSyntaxTreeDataset, metaclass=Singleton):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class NotatedSyntaxValCaptions(CocoNotatedSyntaxTreeDataset, metaclass=Singleton):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class NotatedSyntaxAbsurdCaptions(CocoNotatedSyntaxTreeDataset, metaclass=Singleton):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class ValDetection(torchvision.datasets.CocoDetection, metaclass=Singleton):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class TrainDetection(torchvision.datasets.CocoDetection, metaclass=Singleton):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class AbsurdDetection(torchvision.datasets.CocoDetection, metaclass=Singleton):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
