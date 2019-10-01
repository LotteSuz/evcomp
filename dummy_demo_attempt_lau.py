from controller import Controller
import numpy as np
import time

class controller_automata(Controller, rulebook):

    def __init__(self):
        self.rulebook = rulebook

    def create_random_rulebook(self):
        

    def control(self,inputs,controller):
        # binarize inputs : 
        
