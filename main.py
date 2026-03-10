import hydra
from omegaconf import DictConfig

import jax

from src.system import * 
from src.ansatz import DeepSetsNN
from src.train import run_vmc


@hydra.main(version_base=None, config_path=".", config_name="configs/train")
def main(cfg): 

    ## training for bosons 
    print("lol")









if __name__ == "__main__" : 
    main()