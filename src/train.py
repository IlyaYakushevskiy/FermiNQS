## all the optimiser routines, sampler and logging 
import jax 
import jax.numpy as jnp
import netket as nk
import logging

class Trainer(): 

    def __init__(
            self,
            sampler,
            hamiltonian,  #: System,
            model, #: Gaussian,
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
        vstate = nk.vqs.MCState(self.sampler, self.model, n_samples=10**4, n_discard_per_chain=100)
        #vstate.init_parameters(normal(stddev=1.0))
        optimizer = nk.optimizer.Sgd(learning_rate= self.lr )

        gs_driver = nk.driver.VMC( self.hamiltonian, optimizer= optimizer, variational_state= vstate )

        self.log.info("running driver and logging...")

        nk_log = nk.logging.JsonLog("optimization_results", save_params=True)
        #essentially main loop, computes gradients, feeds to optimizer, computes vstate
        gs_driver.run( n_iter= self.vmc_iters ,out = nk_log)

        self.eigenE = vstate.expect(self.hamiltonian )

        energy_mean = self.eigenE.mean.real
        mc_error = self.eigenE.error_of_mean

        self.log.info(f"Optimized energy and relative error: {energy_mean} ± {mc_error}")

       
        #train loop 

        #self.logger.info("============ Starting epoch %i ... ============" % epoch)

    #def save_model(self, path):

    #compare with analytic solution, implement later 
    #def test(): 

# class Sampler(): 
#     def __init__(): 

#         sampler = nk.sampler.MetropolisGaussian(self.system.hi, sigma=0.1, n_chains=16, sweep_size=32) ##to make variables 