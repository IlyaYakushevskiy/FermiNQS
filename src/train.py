## all the optimiser routines, sampler and logging 
from system import System
from ansatz import Gaussian, DeepSetsNN


import jax 
import jax.numpy as jnp
import netket as nk
import logging

class Trainer(): 

    def __init__(
            self,
            sampler,
            system,  #: System,
            model, #: Gaussian,
            lr : float,
            vmc_iters : int
            
        ): 
        
        self.sampler = sampler 
        self.lr = lr
        self.vmc_iters = vmc_iters
        self.eigenE = None
        self.system = system
        self.model = model

    def __call__(self): 
        

        vstate = nk.vqs.MCState(self.sampler, self.model, n_samples=10**4, n_discard_per_chain=100)
        #vstate.init_parameters(normal(stddev=1.0))
        optimizer = nk.optimizer.Sgd(learning_rate= self.lr )

        gs_driver = nk.driver.VMC( self.system.H, optimizer= optimizer, variational_state= vstate )

        print("running driver and logging...")

        log = nk.logging.RuntimeLog()
        #essentially main loop, computes gradients, feeds to optimizer, computes vstate
        gs_driver.run( n_iter= self.vmc_iters ,out = log)

        self.eigenE = vstate.expect(self.system.H )
        error = jnp.abs(self.eigenE )
        print("Optimized energy and relative error: ", self.eigenE, error)

       
        #train loop 

        #self.logger.info("============ Starting epoch %i ... ============" % epoch)

    #def save_model(self, path):

    #compare with analytic solution, implement later 
    #def test(): 

# class Sampler(): 
#     def __init__(): 

#         sampler = nk.sampler.MetropolisGaussian(self.system.hi, sigma=0.1, n_chains=16, sweep_size=32) ##to make variables 