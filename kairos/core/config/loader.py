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
        if "port" in model.keys():
            model_data['port'] = model['port']
        else:
            port = api_port + port_counter
            port_counter += 1
        relation = model['relation']
        model_id = model['model_id']
        gpu_factor = model['gpu_factor']
        model_data['model_id'] = model_id
        model_data['relation'] = relation
        model_data['storage_location'] = "disk"
        model_data['gpu_factor'] = gpu_factor

        # retrieve data items parsed from huggingface
        meta_data = extract_model_metadata(model_id, relation)

        # unify data items
        model_data.update(meta_data)

        # build model and add to catalog
        build_model_from_config(model_data)


def parse_objectives_from_config(config_path: Path):
    accuracy = None
    latency = None
    with open(config_path, "r") as f:
        registration_data = yaml.safe_load(f)
    service_level_objectives = registration_data.get("slo")
    if "accuracy" in service_level_objectives.keys():
        accuracy = service_level_objectives["accuracy"]
    if "latency" in service_level_objectives.keys():
        latency = service_level_objectives["latency"]
    weight_accuracy = service_level_objectives.get("weight_acc")
    weight_latency = service_level_objectives.get("weight_lat")

    return accuracy, latency, weight_accuracy, latency_accuracy


def parse_monitoring_from_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        registration_data = yaml.safe_load(f)
    monitoring_knobs = registration_data.get("monitoring")
    return monitoring_knobs


def parse_scheduling_from_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        registration_data = yaml.safe_load(f)
    scheduling = registration_data.get("scheduler")
    return scheduling






