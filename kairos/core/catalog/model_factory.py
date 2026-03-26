from kairos.core.catalog.model import Model
from kairos.core.catalog.model_variants_catalog import ModelVariantsCatalog
from kairos.logger import init_logger

logger = init_logger(__name__)


def build_model_from_config(model_data: dict) -> None:
    logger.info(f"Adding model {model_data['name']} to Catalog.")
    model = Model(**model_data)
    catalog = ModelVariantsCatalog()
    catalog.add_model(model)

