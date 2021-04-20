
import argparse
import yaml

class YamlNamespace(argparse.Namespace):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [YamlNamespace(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, YamlNamespace(b) if isinstance(b, dict) else b)

def _parse_args():
    parser = argparse.ArgumentParser("Training script")
    parser.add_argument("--config", "-c", type=str, required=True, help="The YAML config file")
    cli_args = parser.parse_args()
    with open(cli_args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = YamlNamespace(config)
    return config
