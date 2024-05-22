from abc import ABC
from typing import Optional, Callable, Any, List, Dict
import json
import pandas as pd

class BaseDataset(ABC):
    fixed_schema: bool = False
    generator: Optional[Callable[[Any], Any]] = None

    def __init__(
        self,
        fixed_schema: bool = False,
        generator: Optional[Callable[[Any], Any]] = None,
    ):
        self.fixed_schema = fixed_schema
        self.generator = generator

class StaticDataset(BaseDataset):
    file_data: List[Dict[str, Any]] = []

    def __init__(self, data: List[Dict[str, Any]] = []):
        self.file_data = data
        super().__init__(
            fixed_schema=True,
            generator=lambda index: self.file_data[index]
        )

    def from_json(self, file_path: str):
        json_data = None
        with open(file_path) as f:
            json_data = json.load(f)
        self.file_data = json_data
        # check if any keys are inconsistent across records
        keys = [set(record.keys()) for record in self.file_data]
        if len(set(map(len, keys))) > 1:
            self.fixed_schema = False

    def from_csv(self, file_path: str):
        csv_data = pd.read_csv(file_path)
        self.file_data = csv_data.to_dict(orient='records')

class DynamicDataset(BaseDataset):
    def __init__(self, generator: Optional[Callable[[Any], Any]] = None):
        # default generator does a simple lambda return
        # technically collapses into the static case
        if generator is None:
            raise ValueError("DynamicDataset requires a generator function")
        self.generator = generator 
        super().__init__(
            fixed_schema=False,
            generator=generator
        )


__all__ = [
    "BaseDataset",
    "StaticDataset",
    "DynamicDataset",
]
