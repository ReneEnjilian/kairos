from model import Model


class ModelVariantsCatalog:
    def __init__(self):
        self.catalog: dict[str, Model] = {}

    def add_model(self, model: Model):
        self.catalog[model.name] = model
