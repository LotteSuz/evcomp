import spotpy
from A2 import Algoritme2


class spotpy_setup(object):
  def __init__(self):

    self.cmfmodel = Algoritme2()
    self.params = [ 
        #n_point_mut, mutation, number_agents, number_gen
        spotpy.parameter.Uniform('PointMutations', 1, 266),
        spotpy.parameter.Uniform('mutation', 0, 1),
        spotpy.parameter.Uniform('popsize', 10, 50),
        spotpy.parameter.Uniform('generations', 10, 50)
    ]

  def parameters(self):
    return spotpy.parameter.generate(self.params)

  def simulation(self, vector):
    simulations = self.cmfmodel.evolve(vector=vector)
    return simulations

  def evaluation(self):
    #         return self.cmfmodel.obtain_obs_vals(k_shrub = 0.65, k_urban = 0.8)
    #         return self.cmfmodel.get_usgs_fsh()
    return [200]

  def objectivefunction(self, simulation, evaluation):
    objectivefunction = -spotpy.objectivefunctions.rmse(evaluation, simulation)
    return objectivefunction

spotpy_setup = spotpy_setup()

sampler = spotpy.algorithms.rope(spotpy_setup, dbname='fdsjk', dbformat='csv')
results = []
sampler.sample(100)
results.append(sampler.getdata)
