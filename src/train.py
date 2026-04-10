## all the optimiser routines, sampler and logging 
import jax 
import jax.numpy as jnp
import netket as nk
import logging
import wandb
import os
import flax 
import json

class Trainer(): 

    def __init__(
            self,
            sampler,
            hamiltonian,  #: System
            model, #: NQS ansatz
            lr : float,
            vmc_iters : int,
            log : logging.Logger, 
            n_samples : int,
            log_path, 
            pretrained_path,
            exact_gs_energy: float = None,
            seed: int = 42
        ): 
        
        self.sampler = sampler 
        self.lr = lr 
        self.vmc_iters = vmc_iters
        self.eigenE = None
        self.hamiltonian = hamiltonian
        self.model = model
        self.log = log 
        self.n_samples = n_samples
        self.log_path = log_path
        self.pretrained_path = pretrained_path 
        self.exact_gs_energy = exact_gs_energy
        self.seed = seed
    

    def __call__(self): 
        
        #currently expects flax object 
        vstate = nk.vqs.MCState(self.sampler, self.model, n_samples= int(self.n_samples) ,seed=self.seed, n_discard_per_chain=100)

        if self.pretrained_path is not None: 
            with open(self.pretrained_path, "rb") as file:
                vstate.variables = flax.serialization.from_bytes(vstate.variables, file.read())
        
        self.log.info("pre-running sampler ...")
        vstate.sample(chain_length=1000) # run sampler before optimising 

        optimizer = nk.optimizer.Sgd(learning_rate= self.lr)

        gs_driver = nk.driver.VMC_SR( self.hamiltonian, optimizer= optimizer, variational_state= vstate, diag_shift=0.05 )

        self.log.info("running driver and logging...")

        #init array of loggers 
        loggers = [nk.logging.JsonLog("optimization_results", save_params=True)]

        if wandb.run is not None:
            loggers.append(LiveWandbLogger(log_filepath=self.log_path, exact_gs_energy=self.exact_gs_energy))

        #essentially main loop, computes gradients, feeds to optimizer, computes vstate
        gs_driver.run( n_iter= self.vmc_iters ,out=loggers)

        self.eigenE = vstate.expect(self.hamiltonian )

        energy_mean = self.eigenE.mean.real
        mc_error = self.eigenE.error_of_mean

        self.log.info(f"Optimized energy and relative error: {energy_mean} ± {mc_error}")

       
#custom wandb logger for live tracking 

class LiveWandbLogger:
    def __init__(self, log_filepath: str, exact_gs_energy: float = None):
        self.log_file = log_filepath
        self.exact_gs_energy = exact_gs_energy

    def __call__(self, step, item, variational_state):
        if not os.path.exists(self.log_file):
            return 

        try:

            with open(self.log_file, 'r') as f:
                data = json.load(f)

            step_metrics = {}

            for category, metrics_dict in data.items():
                for metric_name, values_array in metrics_dict.items():
                    if metric_name == "iters": 
                        continue 
                    
                    step_metrics[f"{category}/{metric_name}"] = values_array[-1]

            if self.exact_gs_energy is not None:
                step_metrics["Energy/Exact_GS"] = self.exact_gs_energy

            if step_metrics:
                wandb.log(step_metrics, step=step)

        except (json.JSONDecodeError, KeyError, IndexError):
            # If NetKet is mid-write and the JSON is temporarily locked/malformed, 
            # safely ignore it. It will catch up on the very next iteration.
            pass