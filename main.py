import hydra
from omegaconf import DictConfig
##from hydra.utils import instantiate # actually, might have problems with JAX

import jax

from src.system import System
from src.ansatz import Gaussian, DeepSetsNN
from src.train import Trainer


@hydra.main(version_base=None, config_path=".", config_name="configs/train")
def main(cfg : DictConfig): 

    print(f"starting experiment with config: {cfg} ")

    system = System(
        N= cfg.system.N, 
        dim= cfg.system.dim, 
        mass = cfg.system.mass
    )
    
    sampler = nk.sampler.MetropolisGaussian(self.system.hi, sigma=0.1, n_chains=16, sweep_size=32) ##to make variables 






if __name__ == "__main__" : 
    main()