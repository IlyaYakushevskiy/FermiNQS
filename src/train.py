import json
import logging
import os

import flax
import netket as nk
import wandb
from plots.plot_wf import plot_wf
from src.system import System
import jax.scipy.sparse.linalg as jsp_linalg
from functools import partial

#for Adam
import optax 


class Trainer:
    def __init__(
        self,
        sampler,
        hamiltonian,
        model,
        system: System,
        lr: float,
        vmc_iters: int,
        log: logging.Logger,
        run_name: str,
        chunk_size : int,
        lr_decay_rate : float = 1.0, 
        n_samples: int = 10_000,
        log_path: str | None = None,
        pretrained_path: str | None = None,
        diag_shift: float = 0.05,
        n_discard_per_chain: int = 100,
        exact_gs_energy: float | None = None,
        seed: int = 42,
        momentum_beta: float = 0.9,
        optimizer: str = "sgd",
        validation: bool = False, 
        pinv_rtol: float =  1e-14, 
        pinv_atol: float = 1e-14,
        lr_decay_steps: int = 100

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
        self.n_discard_per_chain = n_discard_per_chain
        self.diag_shift = diag_shift
        self.pretrained_path = pretrained_path
        self.exact_gs_energy = exact_gs_energy
        self.momentum_beta = momentum_beta
        self.optimizer = optimizer
        self.seed = seed
        self.validation = validation
        self.run_name = run_name 
        self.system = system
        self.chunk_size = chunk_size
        self.lr_decay_rate = lr_decay_rate
        self.pinv_rtol = pinv_rtol
        self.pinv_atol = pinv_atol
        self.lr_decay_steps = lr_decay_steps
        
    def validation_callback(self, step: int , log_data : dict, driver : nk.driver.AbstractVariationalDriver) -> bool: 
        # E.g., extracts "outputs/2026-04-27/17-17-34"
        working_dir = os.path.dirname(self.log_path)
        ckpt_dir = os.path.join(working_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        if step % 50 == 0: 
            ckpt_filename = os.path.join(ckpt_dir, f"step_{step}.mpack")

            vstate = driver.state
            with open(ckpt_filename, "wb") as f:
                f.write(flax.serialization.to_bytes(vstate.variables))

            #energy check on the copy of a refreshed sampler

            self.log.info(f"running validation at step {step}...")
            
            val_state = nk.vqs.MCState(
                sampler=driver.state.sampler,
                model=driver.state.model,
                n_samples=self.n_samples , 
                chunk_size=driver.state.chunk_size,
                seed=self.seed + step 
            )
            
            val_state.variables = driver.state.variables
            val_energy_stats = val_state.expect(self.hamiltonian)
            
            self.log.info(f"Validation Energy: {val_energy_stats}")
            log_data["Validation_Energy"] = val_energy_stats 
            #plotting
            # plot_path = os.path.join(working_dir, "plots") # no / needed 
            # os.makedirs(plot_path, exist_ok=True)
            # plot_name = f"validation_step_{step}"
            # plot_title = f" {self.run_name}, validation of step {step} with validation energy {val_energy_stats}"
            # plot_wf( plot_name = plot_name, plot_path= plot_path, plot_title= plot_title, system = self.system, vstate = vstate)

            
            
            self.log.info(f"Checkpoint saved to: {ckpt_filename}")

        return True

    def __call__(self):
        vstate = nk.vqs.MCState(
            self.sampler,
            self.model,
            n_samples=int(self.n_samples),
            seed=self.seed,
            n_discard_per_chain=self.n_discard_per_chain,
            chunk_size= self.chunk_size
        )


        if self.pretrained_path is not None:
            with open(self.pretrained_path, "rb") as file:
                vstate.variables = flax.serialization.from_bytes(vstate.variables, file.read())

        #preburning
        # lr_schedule = optax.join_schedules(
        #     schedules=[
        #         optax.constant_schedule(self.lr),
        #         optax.exponential_decay(
        #             init_value=self.lr, 
        #             transition_steps=self.vmc_iters // 150, 
        #             decay_rate=self.lr_decay_rate                    
        #         )
        #     ],
        #     boundaries=[0]
        # )

        lr_schedule = optax.exponential_decay(
            init_value=self.lr, 
            transition_steps=self.vmc_iters // self.lr_decay_steps, # e.g., drop LR every 10% of total iterations
            decay_rate= self.lr_decay_rate                    
        )


        if self.optimizer == "sgd":
            optimizer = nk.optimizer.Sgd(learning_rate=self.lr)
        elif self.optimizer == "momentum":
            optimizer = nk.optimizer.Momentum(learning_rate=lr_schedule, beta=self.momentum_beta)
            self.log.info("Using following learning rate schedule: ", lr_schedule )

        elif self.optimizer == "adam":
            self.log.info("Starting with Adam optimiser")
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),        # grad clipping in nodes 
                optax.adam(learning_rate=lr_schedule) 
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        #Quick experiment, change for adam 

        if self.optimizer == "adam":
            gs_driver = nk.driver.VMC(
            hamiltonian=self.hamiltonian,
            variational_state=vstate,
            optimizer=optimizer,
        )
        else: 
            gs_driver = nk.driver.VMC_SR(
                hamiltonian= self.hamiltonian,
                variational_state = vstate,
                optimizer= optimizer,
                diag_shift=self.diag_shift,
                linear_solver=nk.optimizer.solver.pinv_smooth( rtol = self.pinv_rtol, rtol_smooth = self.pinv_rtol ), ##default values are rtol: float = 1e-14, rtol_smooth: float = 1e-14,
                use_ntk=True, #uses kernel trick (min SR )                        
                mode="complex",                    
                chunk_size_bwd= self.chunk_size
            )
        # driver expects callbacks of form callback: CallbackT | Iterable[CallbackT] = lambda *x: True,

        self.log.info("running driver and logging...")

        loggers = [nk.logging.JsonLog("optimization_results", save_params=True)]

        if wandb.run is not None and self.log_path is not None:
            loggers.append(LiveWandbLogger( exact_gs_energy=self.exact_gs_energy))

        if self.validation == True: 
            gs_driver.run(n_iter=self.vmc_iters, out=loggers, callback= self.validation_callback)
        else: 
            gs_driver.run(n_iter=self.vmc_iters, out=loggers)

        self.eigenE = vstate.expect(self.hamiltonian)

        energy_mean = self.eigenE.mean.real
        mc_error = self.eigenE.error_of_mean

        self.log.info(f"Optimized energy and relative error: {energy_mean} ± {mc_error}")

      


class LiveWandbLogger:
    def __init__(self, exact_gs_energy: float | None = None):
        self.exact_gs_energy = exact_gs_energy

    def __call__(self, step, item, variational_state):
        step_metrics = {}

        for category, value in item.items():
            
            value_dict = value.to_dict() if hasattr(value, "to_dict") else value
            
            if isinstance(value_dict, dict):
                for metric_name, val in value_dict.items():
                    if hasattr(val, "real"):
                        val = val.real
                    step_metrics[f"{category}/{metric_name}"] = val
            else:
                if hasattr(value_dict, "real"):
                    value_dict = value_dict.real
                step_metrics[category] = value_dict

        if self.exact_gs_energy is not None:
            step_metrics["Energy/Exact_GS"] = self.exact_gs_energy

        if step_metrics:
            wandb.log(step_metrics, step=step)

    def flush(self, variational_state):
        pass