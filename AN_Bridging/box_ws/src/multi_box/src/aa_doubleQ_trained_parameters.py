import torch

agents = {}

policyPars = {
    'neurons': (10, 250, 250, 250, 8),  # 5 box-related state, 4 goal-related state, 8 controls
    'act': ['F.leaky_relu', 'F.leaky_relu', 'F.leaky_relu'],
    'mu': torch.zeros(10),
    'std': torch.ones(10),
    'trainMode': False,  # Make sure both value and policy are set to the same thing
    'load': False,
    'dual': False,
    'beta': 12  # boltzmann. Increase for more certainty when making decisions
}
policyTrain = {
    'batch': 256,  # used to be 256
    'lr': 3e-4,
    'buffer': 5000,
    'gamma': .975,
    'explore': 0,  # Don't change this. Tune the exploration up top
    'double': True,
    'noise': 0,
    'priority': False
}
doubleQPars = {'valPars': policyPars, 'valTrain': policyTrain, 'agents': agents}