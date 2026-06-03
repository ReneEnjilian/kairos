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
        self.catalog[model.model_id] = model

    def get_model(self, model_id: str) -> Model:
        return self.catalog.get(model_id)

    def get_base(self) -> Model:
        for model in self.catalog.values():
            if model.relation == "base":
                return model

    def get_catalog(self) -> dict[str, Model]:
        return self.catalog


