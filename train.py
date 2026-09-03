import torch
import model
import yaml
from utils import argparser
from accelerate import utils
import os

### Set random seeds for reproducibility ###
os.environ["PYTHONHASHSEED"] = "8888"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # CUDA determinism for matmul (needed for exact repeatability)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
torch.use_deterministic_algorithms(True, warn_only=True)
###

def main(configs):
    CD_framework=model.Change_Detection_Framework(config=configs)
    CD_framework.training_CD()

if __name__=="__main__":
    utils.set_seed(8888)

    args=argparser.get_argparser().parse_args()

    with open(args.config,'r') as f:
        configs=yaml.safe_load(f)
        print(configs)
    main(configs)
