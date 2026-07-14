import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig, OmegaConf
from hydra.utils import get_original_cwd
import os
import wandb
import itertools

import jax
from netket.utils import struct
import jax.numpy as jnp
import netket as nk
from flax import nnx 
import logging
from netket.sampler.rules import MetropolisRule # TODO create file sampler 


from src.system import System
from src.ansatz import Gaussian, DeepSetsNN, FermiSets, GaussianFermions
from src.train import Trainer
from plots.plot_errs import plot_err

log = logging.getLogger(__name__)

#jax.config.update("jax_debug_nans", True)

# guard against silently-ignored config typos (e.g. the historical 'optmizer' key,
# which made runs fall back to the default sgd without any warning)
ALLOWED_CFG_KEYS = {
    "system": {"N", "dim", "mass", "potential"},
    "ansatz": {"model", "pretrained_path", "hidden_units", "out_units", "pool_fct_name", "L", "lz_proj_K"},
    "sampler": {"sigma", "n_chains", "sweep_size", "exchange_prob", "tune_sigma"},
    "trainer": {
        "lr", "vmc_iters", "n_samples", "diag_shift", "n_discard_per_chain",
        "momentum_beta", "optimizer", "validation", "chunk_size",
        "lr_decay_rate", "lr_decay_steps", "pinv_rtol", "pinv_atol",
        "auto_rollback", "rollback_margin", "max_retries",
    },
}


def validate_config(cfg: DictConfig):
    for section, allowed in ALLOWED_CFG_KEYS.items():
        if section not in cfg:
            continue
        unknown = set(cfg[section].keys()) - allowed
        if unknown:
            raise ValueError(
                f"Unknown key(s) {sorted(unknown)} in cfg.{section} — typo? "
                f"Allowed: {sorted(allowed)}"
            )


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

# TODO move into separate file 
class SamplerExchangeRule(MetropolisRule):
    """
    A custom rule for spinless fermions: interweaves Gaussian drift 
    with uniform particle exchanges across the entire system.
    """

    sigma: float
    exchange_prob: float
    n_particles: int = struct.field(pytree_node=False)
    n_dim: int = struct.field(pytree_node=False)

    
    def __init__(self, sigma: float, exchange_prob: float, n_particles: int, n_dim: int): 
        self.sigma = sigma 
        self.exchange_prob = exchange_prob 
        self.n_particles = n_particles
        self.n_dim = n_dim

    def transition(self, sampler, machine, parameters, state, key, r):
        n_chains = r.shape[0]
        key_action, key_gauss, key_i, key_j = jax.random.split(key, 4)
        
        # decide operation (True = Exchange, False = Gaussian)
        do_exchange = jax.random.bernoulli(key_action, p=self.exchange_prob, shape=(n_chains, 1))
        
        # PROPOSAL A: Gaussian Drift 
        r_gaussian = r + self.sigma * jax.random.normal(key_gauss, shape=r.shape)
        
        # PROPOSAL B: Universal Exchange 
        r_reshaped = r.reshape(n_chains, self.n_particles, self.n_dim)
        
        # sample two DISTINCT particle indices (i == j would waste the move on an identity swap)
        idx_i = jax.random.randint(key_i, shape=(n_chains,), minval=0, maxval=self.n_particles)
        offset = jax.random.randint(key_j, shape=(n_chains,), minval=1, maxval=self.n_particles)
        idx_j = (idx_i + offset) % self.n_particles
        
        # extract the particles
        particle_i = r_reshaped[jnp.arange(n_chains), idx_i, :]
        particle_j = r_reshaped[jnp.arange(n_chains), idx_j, :]
        

        r_swapped = r_reshaped.at[jnp.arange(n_chains), idx_i, :].set(particle_j)
        r_swapped = r_swapped.at[jnp.arange(n_chains), idx_j, :].set(particle_i)
        #flatten
        r_exchange = r_swapped.reshape(n_chains, self.n_particles * self.n_dim)
        
        r_proposed = jnp.where(do_exchange, r_exchange, r_gaussian)
        log_prob_correction = jnp.zeros(n_chains)
        
        return r_proposed, log_prob_correction
    

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg : DictConfig): 

    validate_config(cfg)

    hydra_cfg = HydraConfig.get()
    current_out_dir = hydra_cfg.runtime.output_dir
    time_stamp = os.path.basename(current_out_dir)
    run_name = f"{cfg.system.potential}_{cfg.ansatz.model}_N{cfg.system.N}_{time_stamp}"
    orig_cwd = get_original_cwd()

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
        fermi_hidden_units = cfg.ansatz.get("hidden_units", 8)
        fermi_out_units = cfg.ansatz.get("out_units", 10)
        fermi_pool_fct_name = cfg.ansatz.get("pool_fct_name", None)
        fermi_L = cfg.ansatz.get("L", None)

        ansatz = FermiSets(
            dim= cfg.system.dim,
            rngs= nnx.Rngs(42),
            N = cfg.system.N,
            hidden_units= fermi_hidden_units,
            out_units = fermi_out_units,
            pool_fct_name=fermi_pool_fct_name,
            L=fermi_L,
            log= log,
            lz_proj_K = cfg.ansatz.get("lz_proj_K", 0)
        )

        #not very usefull, delete mb 
        if wandb.run is not None:
            wandb.config.update(
                {
                    "ansatz/hidden_units_effective": fermi_hidden_units,
                    "ansatz/out_units_effective": fermi_out_units,
                    "ansatz/pool_fct_name_effective": fermi_pool_fct_name,
                    "ansatz/L_effective": fermi_L,
                },
                allow_val_change=True,
            )

    if cfg.ansatz.model ==  "gaussian_fermions": 
        ansatz = GaussianFermions(
            dim= cfg.system.dim,
            rngs= nnx.Rngs(42),
            N = cfg.system.N
        )

    
    custom_rule = SamplerExchangeRule(
        sigma=cfg.sampler.sigma,
        exchange_prob=cfg.sampler.get("exchange_prob", 0.1),
        n_particles=system.N,
        n_dim=cfg.system.dim
    )

    sampler = nk.sampler.MetropolisSampler(
        hilbert=system.hi,
        rule=custom_rule,
        n_chains=cfg.sampler.n_chains,
        sweep_size=cfg.sampler.sweep_size
    )
    # sampler = nk.sampler.MetropolisGaussian(system.hi, 
    #                                         sigma=cfg.sampler.sigma,
    #                                         n_chains=cfg.sampler.n_chains,
    #                                         sweep_size=cfg.sampler.sweep_size) ##to make variables 

    
    trainer = Trainer(
        sampler=sampler,
        hamiltonian=system.H,
        model=ansatz,
        chunk_size= cfg.trainer.chunk_size, 
        lr=cfg.trainer.lr,
        vmc_iters=cfg.trainer.vmc_iters,
        log=log,
        n_samples=cfg.trainer.n_samples,
        log_path=log_path,
        pretrained_path=cfg.ansatz.pretrained_path,
        diag_shift=cfg.trainer.diag_shift,
        n_discard_per_chain=cfg.trainer.n_discard_per_chain,
        exact_gs_energy=exact_energy,
        seed=cfg.get("seed", 42),
        momentum_beta=cfg.trainer.momentum_beta,
        optimizer=cfg.trainer.optimizer,
        validation= cfg.trainer.validation, 
        run_name = run_name,
        system = system,
        lr_decay_rate = cfg.trainer.lr_decay_rate,
        lr_decay_steps = cfg.trainer.lr_decay_steps,
        pinv_rtol = cfg.trainer.pinv_rtol,
        pinv_atol = cfg.trainer.pinv_atol,
        tune_sigma = cfg.sampler.get("tune_sigma", True),
        auto_rollback = cfg.trainer.get("auto_rollback", True),
        rollback_margin = cfg.trainer.get("rollback_margin", 2.0),
        max_retries = cfg.trainer.get("max_retries", 2)
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