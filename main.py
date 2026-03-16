import hydra
from omegaconf import DictConfig
##from hydra.utils import instantiate # actually, might have problems with JAX

import jax
import netket as nk
from flax import nnx 
import logging

from src.system import System
from src.ansatz import Gaussian, DeepSetsNN
from src.train import Trainer

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg : DictConfig): 

    log.info(f"starting experiment with config: {cfg} ")

    system = System(
        N= cfg.system.N, 
        dim= cfg.system.dim, 
        mass = cfg.system.mass,
        potential= cfg.system.potential
    )

    if cfg.ansatz.model ==  "gaussian":

        ansatz = Gaussian(
            dim= cfg.system.dim,
            rngs= nnx.Rngs(42),
            N = cfg.system.N
        )

    if cfg.ansatz.model ==  "deep_sets": 
        ansatz = DeepSetsNN(
            dim= cfg.system.dim,
            rngs= nnx.Rngs(42),
            N = cfg.system.N
        )
    
    sampler = nk.sampler.MetropolisGaussian(system.hi, 
                                            sigma=0.1,
                                            n_chains=16,
                                            sweep_size=32) ##to make variables 

    trainer = Trainer(
     sampler=sampler,
     hamiltonian = system.H, 
     model = ansatz,
     lr = cfg.trainer.lr,
     vmc_iters= cfg.trainer.vmc_iters,
     log = log
    )
    
    trainer()


if __name__ == "__main__" : 
    main()