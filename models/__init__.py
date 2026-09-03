import importlib

def find_model_using_name(configs):
    model_name=configs["model_name"]
    print(f"loading {model_name}")
    if model_name == "CMD2Net":
        module=importlib.import_module(f'.CMD2Net.model', package='models')
        model=module.CMD2Net()

    return model
