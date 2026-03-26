from huggingface_hub import HfApi, HfFileSystem

from typing import Any, Dict
import json

hf_api = HfApi()
hf_file_system = HfFileSystem()


def get_model_size_bytes(repo_id: str) -> int | None:
    info = hf_api.model_info(repo_id, files_metadata=True)

    safetensors = [
        f for f in (info.siblings or [])
        if f.rfilename.endswith(".safetensors")
    ]

    sizes = [f.size for f in safetensors if f.size is not None]
    return sum(sizes) if sizes else None


def extract_model_metadata(model_id: str, relation: str) -> Dict[Any, Any]:
    meta_data: dict[Any, Any] = {}
    info = hf_api.model_info(
        model_id,
        expand=["baseModels", "cardData", "config", "tags", "siblings"],

    )

    # extract content of config.json file
    config_file = json.loads(hf_file_system.read_text(model_id + "/config.json"))

    if relation == "quantized":
        # get original base model
        base_model = info.base_models.get("models")[0]
        lineage = base_model['id']
        meta_data['lineage'] = lineage

        # get quant_method: bitsandbytes, gptq, compressed-tensors etc.
        quant_method = info.config['quantization_config']['quant_method']
        meta_data['quant_method'] = quant_method

        # extract quantization configs from config.json
        quantization_config = config_file['quantization_config']

        if quant_method == "compressed-tensors":
            input_activations = quantization_config['config_groups']['group_0']['input_activations']
            weights = quantization_config['config_groups']['group_0']['weights']

            # some models do not quantize activations
            if input_activations:
                activation_bits = input_activations['num_bits']
                meta_data['activation_bits'] = activation_bits
                activation_type = input_activations['type']
                meta_data['activation_type'] = activation_type
            weight_bits = weights['num_bits']
            meta_data['weight_bits'] = weight_bits
            weight_type = weights['type']
            meta_data['weight_type'] = weight_type

        # bitsandbytes does not quantize activations
        if quant_method == "bitsandbytes":
            weight_bits = 4 if quantization_config['_load_in_4bit'] else 8
            meta_data['weight_bits'] = weight_bits
            weight_type = quantization_config['bnb_4bit_quant_type']
            meta_data['weight_type'] = weight_type

        # gptq is weight-only quantization
        if quant_method == "gptq":
            weight_bits = quantization_config['bits']
            meta_data['weight_bits'] = weight_bits
            meta_data['weight_type'] = 'int'

    # get tensor type
    compute_dtype = config_file.get("dtype") or config_file.get("torch_dtype")
    meta_data['compute_dtype'] = compute_dtype

    # calculate model size in bytes based on safetensors
    size_bytes = get_model_size_bytes(model_id)
    meta_data['size'] = size_bytes

    return meta_data
