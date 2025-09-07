import json
def get_configs(task_name):
    with open(f"configs/{task_name}_config.json") as f:
        config = json.load(f)
        return config