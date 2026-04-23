import json
import logging
import os

import flax
import netket as nk
import wandb


class Trainer:
    def __init__(
        self,
        sampler,
        hamiltonian,
        model,
        lr: float,
        vmc_iters: int,
        log: logging.Logger,
        n_samples: int = 10_000,
        log_path: str | None = None,
        pretrained_path: str | None = None,
        diag_shift: float = 0.05,
        n_discard_per_chain: int = 100,
        exact_gs_energy: float | None = None,
        seed: int = 42,
        momentum_beta: float = 0.9,
        optimizer: str = "sgd",
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

    def __call__(self):
        vstate = nk.vqs.MCState(
            self.sampler,
            self.model,
            n_samples=int(self.n_samples),
            seed=self.seed,
            n_discard_per_chain=self.n_discard_per_chain,
        )

        if self.pretrained_path is not None:
            with open(self.pretrained_path, "rb") as file:
                vstate.variables = flax.serialization.from_bytes(vstate.variables, file.read())

        if self.optimizer == "sgd":
            optimizer = nk.optimizer.Sgd(learning_rate=self.lr)
        elif self.optimizer == "momentum":
            optimizer = nk.optimizer.Momentum(learning_rate=self.lr, beta=self.momentum_beta)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        gs_driver = nk.driver.VMC_SR(
            self.hamiltonian,
            optimizer=optimizer,
            variational_state=vstate,
            diag_shift=self.diag_shift,
        )

        self.log.info("running driver and logging...")

        loggers = [nk.logging.JsonLog("optimization_results", save_params=True)]

        if wandb.run is not None and self.log_path is not None:
            loggers.append(LiveWandbLogger(log_filepath=self.log_path, exact_gs_energy=self.exact_gs_energy))

        gs_driver.run(n_iter=self.vmc_iters, out=loggers)

        self.eigenE = vstate.expect(self.hamiltonian)

        energy_mean = self.eigenE.mean.real
        mc_error = self.eigenE.error_of_mean

        self.log.info(f"Optimized energy and relative error: {energy_mean} ± {mc_error}")


class LiveWandbLogger:
    def __init__(self, log_filepath: str, exact_gs_energy: float | None = None):
        self.log_file = log_filepath
        self.exact_gs_energy = exact_gs_energy

    def __call__(self, step, item, variational_state):
        if not os.path.exists(self.log_file):
            return

        try:
            with open(self.log_file, "r") as f:
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
            pass