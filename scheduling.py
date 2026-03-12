# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 08:54:46 2026

@author: Nicc
"""

from parameter_manager import ParameterManager

class Scheduling(ParameterManager):
    
    def __init__(
            self,
            trials=10,
            timesteps=1000,
            phases=(0.25, 0.5, 0.75, 1.0),
            states=((0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
                 (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 (1.0, 1.0, 0.0, 0.0, 1.0, 0.0),
                 (1.0, 1.0, 0.0, 0.0, 1.0, 0.0)
                 )
            ):
        
        self.trials=trials
        self.timesteps=timesteps
        self.phases=phases
        self.states=states
        
        super(Scheduling, self).__init__()
        