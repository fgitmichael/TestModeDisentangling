from test import mode_diversity_test
from env import OrdinaryEnvForPytorch

if __name__ == '__main__':
    env = OrdinaryEnvForPytorch(gym_id='MountainCarContinuous-v0')
    model_path = './logs/disentangling/cheetah-run/' + \
                 'slac-seed0-20200601-2255/model/slac-seed0-20200601-2255whole_model.pth'
    tester = mode_diversity_test.DisentanglingDiversityTester(latent_model_path=model_path,
                                 env=env)
    tester.run()

