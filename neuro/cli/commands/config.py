"""Config command."""

import os
import json


def cmd_config(args):
    """Open or display configuration."""
    config_path = os.path.expanduser("~/.neuro/settings.json")

    print(f"\n  Configuration")
    print(f"  {'─' * 50}")
    print(f"  Config file: {config_path}")

    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            print(f"\n  Current settings:")
            for k, v in cfg.items():
                print(f"    {k}: {v}")
        except Exception as e:
            print(f"  Error reading config: {e}")
    else:
        print(f"  (Config file not found - using defaults)")
        print(f"\n  Create one with:")
        print(f'    mkdir -p ~/.neuro')
        print(f'    echo \'{{"model": "ministral-3:8b"}}\' > {config_path}')

    print()
    return 0
