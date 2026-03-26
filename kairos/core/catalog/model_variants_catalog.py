from __future__ import annotations
from kairos.core.catalog.model import Model


class ModelVariantsCatalog:

    _instance: ModelVariantsCatalog | None = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self.__class__._initialized:
            return
        self.catalog: dict[str, Model] = {}
        self.__class__._initialized = True

    def add_model(self, model: Model) -> None:
        self.catalog[model.name] = model

    def get_model(self, name: str) -> Model:
        return self.catalog.get(name)
