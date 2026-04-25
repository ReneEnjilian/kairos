import yaml

from pathlib import Path
from kairos.core.config.model_metadata import extract_model_metadata
from kairos.core.catalog.model_factory import build_model_from_config


def parse_models_from_config(config_path: Path, api_port: int) -> None:
    model_data = {}
    with open(config_path, "r") as f:
        registration_data = yaml.safe_load(f)
    port_counter = 1
    for model in registration_data['models']:
        if "name" in model.keys():
            model_data['name'] = model['name']
        else:
            model_data['name'] = model['model_id']

        if "port" in model.keys():
            model_data['port'] = model['port']
        else:
            port = api_port + port_counter
            port_counter += 1
        relation = model['relation']
        model_id = model['model_id']
        model_data['model_id'] = model_id
        model_data['relation'] = relation
        model_data['storage_location'] = "disk"

        # retrieve data items parsed from huggingface
        meta_data = extract_model_metadata(model_id, relation)

        # unify data items
        model_data.update(meta_data)

        # build model and add to catalog
        build_model_from_config(model_data)


# TODO: Implement parsing of SLOs and knobs
def parse_objectives_from_config(config_path: Path):
    accuracy = None
    latency = None
    with open(config_path, "r") as f:
        registration_data = yaml.safe_load(f)
    service_level_objectives = registration_data.get("SLOs")

    for slo in service_level_objectives:
        if "accuracy" in slo.keys():
            accuracy = slo["accuracy"]
        elif "latency" in slo.keys():
            latency = slo["latency"]

    return accuracy, latency


def parse_knobs_from_config(config_path: Path):
    pass



