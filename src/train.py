import glob
import json
import logging
import os

import flax
import netket as nk
import numpy as np
import wandb
from plots.plot_wf import plot_wf
from src.system import System
import jax.scipy.sparse.linalg as jsp_linalg
import jax
from functools import partial
import jax.numpy as jnp

#for Adam
import optax

from src.holomorphy_penalty import HolomorphyPenalty
from src.deflation_penalty import DeflationPenalty


class BlowupGuard:
    """
    Driver callback that stops the run when the energy diverges (NaN or a jump
    far above the best value seen), so the Trainer can roll back to the last
    good checkpoint with a reduced learning rate. Tracks the newest checkpoint
    file that was written while the energy was still sane.
    """

    def __init__(self, log: logging.Logger, ckpt_dir: str | None, margin: float = 2.0):
        self.log = log
        self.ckpt_dir = ckpt_dir
        self.margin = margin
        self.best = None
        self.tripped = False
        self.last_good_ckpt = None

    def _newest_ckpt(self):
        if self.ckpt_dir is None:
            return None
        files = glob.glob(os.path.join(self.ckpt_dir, "step_*.mpack"))
        if not files:
            return None
        return max(files, key=lambda p: int(p.split("_")[-1].split(".")[0]))

    def __call__(self, step: int, log_data: dict, driver) -> bool:
        stats = log_data.get("Energy", None)
        if stats is None:
            return True
        mean = float(np.real(stats.mean)) if hasattr(stats, "mean") else float(np.real(stats))
        var = float(np.real(getattr(stats, "variance", 0.0)))

        threshold = None
        if self.best is not None:
            threshold = self.best + max(self.margin, 5.0 * np.sqrt(max(var, 0.0)))

        if not np.isfinite(mean) or (threshold is not None and mean > threshold):
            self.tripped = True
            self.log.error(
                f"blow-up guard tripped at step {step}: E={mean}, best={self.best}, "
                f"threshold={threshold}"
            )
            return False

        if self.best is None or mean < self.best:
            self.best = mean
        self.last_good_ckpt = self._newest_ckpt()
        return True


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
        lr_decay_steps: int = 100,
        tune_sigma: bool = True,
        auto_rollback: bool = True,
        rollback_margin: float = 2.0,
        max_retries: int = 2,
        holo_penalty: bool = False,
        holo_penalty_mu0: float = 0.0,
        holo_penalty_decay: float = 1.0,
        holo_penalty_lr: float = 0.0,
        deflation_penalty: bool = False,
        deflation_penalty_mu0: float = 0.0,
        deflation_penalty_decay: float = 1.0,
        deflation_penalty_lr: float = 0.0,
        deflation_penalty_n_ginibre: int = 2000,

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
        self.tune_sigma = tune_sigma
        self.auto_rollback = auto_rollback
        self.rollback_margin = rollback_margin
        self.max_retries = max_retries
        self.holo_penalty = holo_penalty
        self.holo_penalty_mu0 = holo_penalty_mu0
        self.holo_penalty_decay = holo_penalty_decay
        self.holo_penalty_lr = holo_penalty_lr
        self.deflation_penalty = deflation_penalty
        self.deflation_penalty_mu0 = deflation_penalty_mu0
        self.deflation_penalty_decay = deflation_penalty_decay
        self.deflation_penalty_lr = deflation_penalty_lr
        self.deflation_penalty_n_ginibre = deflation_penalty_n_ginibre

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

            acc = getattr(driver.state.sampler_state, "acceptance", None)
            if acc is not None:
                self.log.info(f"training sampler acceptance: {float(acc):.3f}")
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

        if self.tune_sigma:
            self._tune_sampler_sigma(vstate)

        ckpt_dir = None
        if self.validation and self.log_path is not None:
            ckpt_dir = os.path.join(os.path.dirname(self.log_path), "checkpoints")

        current_lr = self.lr
        attempt = 0
        while True:
            # optimizer/driver rebuilt on every attempt so momentum state resets after a rollback
            gs_driver = self._build_driver(vstate, current_lr)

            self.log.info("running driver and logging...")
            loggers = [nk.logging.JsonLog("optimization_results", save_params=True)]
            if wandb.run is not None and self.log_path is not None:
                loggers.append(LiveWandbLogger(exact_gs_energy=self.exact_gs_energy))

            guard = BlowupGuard(self.log, ckpt_dir, margin=self.rollback_margin)
            callbacks = [guard]
            if self.validation:
                callbacks.append(self.validation_callback)
            if self.holo_penalty:
                # N*dim inferred from the model's own N, dim attrs (FermiSets, dim=2 only)
                penalty = HolomorphyPenalty(
                    template_model=self.model, N=self.model.N,
                    mu0=self.holo_penalty_mu0, decay=self.holo_penalty_decay,
                    aux_lr=self.holo_penalty_lr, log=self.log,
                )
                callbacks.append(penalty)
            if self.deflation_penalty:
                penalty2 = DeflationPenalty(
                    template_model=self.model, N=self.model.N,
                    n_ginibre=self.deflation_penalty_n_ginibre,
                    mu0=self.deflation_penalty_mu0, decay=self.deflation_penalty_decay,
                    aux_lr=self.deflation_penalty_lr, log=self.log,
                )
                callbacks.append(penalty2)

            gs_driver.run(n_iter=self.vmc_iters, out=loggers, callback=callbacks)

            if (
                self.auto_rollback
                and guard.tripped
                and attempt < self.max_retries
                and guard.last_good_ckpt is not None
            ):
                with open(guard.last_good_ckpt, "rb") as f:
                    vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())
                attempt += 1
                current_lr *= 0.5
                self.log.warning(
                    f"rolled back to {guard.last_good_ckpt}; retry {attempt}/{self.max_retries} "
                    f"with lr={current_lr}"
                )
                continue
            if guard.tripped:
                self.log.error("energy blew up and no rollback was possible; final state is not trustworthy")
            break

        self.eigenE = vstate.expect(self.hamiltonian)

        energy_mean = self.eigenE.mean.real
        mc_error = self.eigenE.error_of_mean

        self.log.info(f"Optimized energy and relative error: {energy_mean} ± {mc_error}")

    def _tune_sampler_sigma(self, vstate, target_low: float = 0.35,
                            target_high: float = 0.6, max_rounds: int = 8):
        """Multiplicatively adjust the drift sigma toward a healthy acceptance window."""
        rule = getattr(vstate.sampler, "rule", None)
        if rule is None or not hasattr(rule, "sigma"):
            self.log.warning("sigma tuning skipped: sampler rule has no sigma")
            return
        sigma = float(rule.sigma)
        for _ in range(max_rounds):
            vstate.sample()
            acc = getattr(vstate.sampler_state, "acceptance", None)
            if acc is None:
                self.log.warning("sigma tuning skipped: sampler exposes no acceptance")
                return
            acc = float(acc)
            self.log.info(f"sigma tuning: sigma={sigma:.4f}, acceptance={acc:.3f}")
            if target_low <= acc <= target_high:
                break
            factor = float(np.clip(acc / 0.5, 0.5, 2.0))
            sigma = float(np.clip(sigma * factor, 1e-3, 3.0))
            try:
                vstate.sampler = vstate.sampler.replace(rule=rule.replace(sigma=sigma))
                rule = vstate.sampler.rule
            except (AttributeError, TypeError) as e:
                self.log.warning(f"sigma tuning aborted, cannot replace sampler rule: {e}")
                return
        self.log.info(f"sampler sigma after tuning: {sigma:.4f}")

    def _build_driver(self, vstate, lr):
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
            init_value=lr,
            transition_steps = self.lr_decay_steps, # e.g., drop LR every 10% of total iterations
            decay_rate= self.lr_decay_rate
        )


        if self.optimizer == "sgd":
            optimizer = nk.optimizer.Sgd(learning_rate=lr_schedule)
        elif self.optimizer == "momentum":
            optimizer = nk.optimizer.Momentum(learning_rate=lr_schedule, beta=self.momentum_beta)
            self.log.info(f"Using following learning rate schedule: {lr_schedule}")

        elif self.optimizer == "adam":
            self.log.info("Starting with Adam optimiser")
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),        # grad clipping in nodes 
                optax.adam(learning_rate=lr_schedule) 
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        #Quick experiment, change for adam 

        def make_safe_solver(base_solver, max_bilinear_form=1000.0):
            """
            Wraps a NetKet linear solver with a trust region on the SR step:
            the bilinear form <x|S|x> = <x|b> is the squared natural-gradient
            norm of the update. If it exceeds `max_bilinear_form`, the step is
            rescaled onto the trust-region boundary (direction preserved)
            instead of being discarded.
            """
            def safe_solver(A, b):
                x, info = base_solver(A, b)

                bilinear_form = jnp.real(nk.jax.tree_dot(x, b))

                scale = jnp.where(
                    bilinear_form > max_bilinear_form,
                    jnp.sqrt(max_bilinear_form / jnp.abs(bilinear_form)),
                    1.0,
                )
                x_safe = jax.tree_util.tree_map(lambda leaf: scale * leaf, x)
                return x_safe, info

            return safe_solver

        if self.optimizer == "adam":
            gs_driver = nk.driver.VMC(
            hamiltonian=self.hamiltonian,
            variational_state=vstate,
            optimizer=optimizer,
        )
        else: 


            # base_solver = nk.optimizer.solver.pinv_smooth(
            #     rtol=self.pinv_rtol, 
            #     rtol_smooth=self.pinv_rtol
            # )

            base_solver = nk.optimizer.solver.cholesky_with_fallback( rtol = self.pinv_rtol, rtol_smooth = self.pinv_rtol )



            safe_solver = make_safe_solver(base_solver, max_bilinear_form=900.0)

            gs_driver = nk.driver.VMC_SR(
                hamiltonian= self.hamiltonian,
                variational_state = vstate,
                optimizer= optimizer,
                diag_shift=self.diag_shift,
                #linear_solver=nk.optimizer.solver.pinv_smooth( rtol = self.pinv_rtol, rtol_smooth = self.pinv_rtol ), ##default values are rtol: float = 1e-14, rtol_smooth: float = 1e-14,
                linear_solver = safe_solver, 
                #linear_solver= nk.optimizer.solver.cholesky_with_fallback( rtol = self.pinv_rtol, rtol_smooth = self.pinv_rtol ),
                use_ntk=True, #uses kernel trick (min SR )                        
                mode="complex",                    
                chunk_size_bwd= self.chunk_size
            )
        return gs_driver

      


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