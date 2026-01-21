import sys
from pathlib import Path
import yaml

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

config_path = repo_root / "main" / "config.yml"

with config_path.open("r", encoding="utf-8") as handle:
    config_dict = yaml.safe_load(handle)
    #print(type(config_dict['structures']['diamond']['file_path']))
    print(type(config_dict['structures']['diamond']['is_file_relaxed']))
    if config_dict['structures']['diamond']['is_file_relaxed']:
        print('here')
        




