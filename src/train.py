## all the optimiser routines, sampler and logging 
import jax 
import jax.numpy as jnp
import netket as nk
import logging
import flax

class Trainer(): 

    def __init__(
            self,
            sampler,
            hamiltonian,  #: System
            model, #: NQS ansatz
            lr : float,
            vmc_iters : int,
            log : logging.Logger
        ): 
        
        self.sampler = sampler 
        self.lr = lr
        self.vmc_iters = vmc_iters
        self.eigenE = None
        self.hamiltonian = hamiltonian
        self.model = model
        self.log = log 

    def __call__(self): 
        
        #currently expects flax object 

        #mpack_path = "/home/ilya/FermiNQS/outputs/2026-04-20/22-18-24/optimization_results.mpack" #to make var
        mpack_path = "/home/ilya/FermiNQS/outputs/2026-04-20/22-53-56/optimization_results.mpack"

        vstate = nk.vqs.MCState(self.sampler, self.model, n_samples=10**4, n_discard_per_chain=1000, seed=42 )

        # if(mpack_path is not None):
        #     with open(mpack_path, "rb") as file:
        #         print("loading from the chekpoint in path")
        #         vstate.variables = flax.serialization.from_bytes(vstate.variables, file.read())

        #vstate.init_parameters(normal(stddev=1.0))
        #optimizer = nk.optimizer.Sgd(learning_rate= self.lr )
        optimizer = nk.optimizer.Momentum(learning_rate=0.005, beta= 0.8)

        gs_driver = nk.driver.VMC_SR( self.hamiltonian, optimizer= optimizer, variational_state= vstate, diag_shift=0.2 )

        self.log.info("running driver and logging...")

        nk_log = nk.logging.JsonLog("optimization_results", save_params=True)
        #essentially main loop, computes gradients, feeds to optimizer, computes vstate
        gs_driver.run( n_iter= self.vmc_iters ,out = nk_log)

        self.eigenE = vstate.expect(self.hamiltonian )

        energy_mean = self.eigenE.mean.real
        mc_error = self.eigenE.error_of_mean

        self.log.info(f"Optimized energy and relative error: {energy_mean} ± {mc_error}")

       
       


