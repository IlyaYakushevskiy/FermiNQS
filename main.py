import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
import os
##from hydra.utils import instantiate # actually, might have problems with JAX

import jax
import netket as nk
from flax import nnx 
import logging


from src.system import System
from src.ansatz import Gaussian, DeepSetsNN, FermiSets, GaussianFermions
from src.train import Trainer
from plots.plot_errs import plot_err

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

    if cfg.ansatz.model ==  "fermi_sets": 
        ansatz = FermiSets(
            dim= cfg.system.dim,
            rngs= nnx.Rngs(42),
            N = cfg.system.N, 
            hidden_units= cfg.ansatz.hidden_units
        )

    if cfg.ansatz.model ==  "gaussian_fermions": 
        ansatz = GaussianFermions(
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

    #Plotting errors 
    hydra_cfg = HydraConfig.get()
    current_out_dir = hydra_cfg.runtime.output_dir
    orig_cwd = get_original_cwd()

    time_stamp = os.path.basename(current_out_dir) 
    run_name = f"{cfg.system.potential}_{cfg.ansatz.model}_N{cfg.system.N}_{time_stamp}"

    log_path = os.path.join(current_out_dir, "optimization_results.log")
    plot_dir = os.path.join(orig_cwd, "plots")

    if os.path.exists(log_path):
        plot_err(log_path=log_path, plot_name=run_name, save_dir=plot_dir)
        log.info(f"plot saved to: {os.path.join(plot_dir, run_name)}.png")
    else:
        log.error(f"could not find log file at {log_path}")

if __name__ == "__main__" : 
    main()