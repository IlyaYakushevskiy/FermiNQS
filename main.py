import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig, OmegaConf
from hydra.utils import get_original_cwd
import os
import wandb
import itertools

import jax
import netket as nk
from flax import nnx 
import logging


from src.system import System
from src.ansatz import Gaussian, DeepSetsNN, FermiSets, GaussianFermions
from src.train import Trainer
from plots.plot_errs import plot_err

log = logging.getLogger(__name__)

def exact_qho_gs_energy(N: int, dim: int, statistics: str = "fermion") -> float:

    base_energy = 0.5 * dim

    if statistics in ["boson", "distinguishable"]:
        return N * base_energy

    elif statistics == "fermion":
        ranges = [range(N + 1)] * dim
        
        state_energies = [sum(quantum_numbers) for quantum_numbers in itertools.product(*ranges)]
        state_energies.sort()
        gs_energy = sum(state_energies[:N]) + (N * base_energy)
        return float(gs_energy)
    else:
        raise ValueError(f"Unknown statistics: {statistics}")


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg : DictConfig): 

    hydra_cfg = HydraConfig.get()
    current_out_dir = hydra_cfg.runtime.output_dir
    orig_cwd = get_original_cwd()

    time_stamp = os.path.basename(current_out_dir) 
    run_name = f"{cfg.system.potential}_{cfg.ansatz.model}_N{cfg.system.N}_{time_stamp}"

    if cfg.get("use_wnb", False): 
        wandb.init(
            project="FermiNQS",
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    log.info(f"starting experiment with config: {cfg} ")
    system = System(
        N= cfg.system.N, 
        dim= cfg.system.dim, 
        mass = cfg.system.mass,
        potential= cfg.system.potential
    )

    is_fermionic = "fermi" in cfg.ansatz.model
    statistics = "fermion" if is_fermionic else "boson"
    
    exact_energy = exact_qho_gs_energy(cfg.system.N, cfg.system.dim, statistics)

    hydra_cfg = HydraConfig.get()
    current_out_dir = hydra_cfg.runtime.output_dir
    log_path = os.path.join(current_out_dir, "optimization_results.log")

    log.info(f"Exact theoretical ground state energy calculated as: {exact_energy}")

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
                                            sigma=cfg.sampler.sigma,
                                            n_chains=cfg.sampler.n_chains,
                                            sweep_size=cfg.sampler.sweep_size) ##to make variables 

    trainer = Trainer(
     sampler=sampler,
     hamiltonian = system.H, 
     model = ansatz,
     lr = cfg.trainer.lr,
     vmc_iters= cfg.trainer.vmc_iters,
     n_samples= cfg.trainer.n_samples,
     log = log,
     log_path = log_path,
     seed=cfg.get("seed", 42),
     exact_gs_energy=exact_energy
    )
    
    trainer()

    #Plotting errors 
    
    orig_cwd = get_original_cwd()

    time_stamp = os.path.basename(current_out_dir) 
    run_name = f"{cfg.system.potential}_{cfg.ansatz.model}_N{cfg.system.N}_{time_stamp}"
    plot_dir = os.path.join(orig_cwd, "plots")

    if os.path.exists(log_path):
        plot_err(log_path=log_path, plot_name=run_name, save_dir=plot_dir)
        log.info(f"plot saved to: {os.path.join(plot_dir, run_name)}.png")
    else:
        log.error(f"could not find log file at {log_path}")
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__" : 
    main()