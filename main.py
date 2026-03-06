import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=".", config_name="configs/train")
def train(): 
    print("lol") 


if __name__ == "__main__" : 
    train()