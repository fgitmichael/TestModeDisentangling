import os
import argparse
from datetime import datetime

from env import DmControlEnvForPytorch, GymEnvForPyTorch, DmControlEnvForPytorchBothObstype
from disentangling_trainer import DisentanglingTrainer

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', type=str, default='cheetah')
    parser.add_argument('--task_name', type=str, default='run')
    parser.add_argument('--env_id', type=str, default='HalfCheetah-v2')
    parser.add_argument('--action_repeat', type=int, default=4)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy_path', type=str, default=\
        './data/01_Skill_policy_half_cheetah/params.pkl')
    args = parser.parse_args()

    configs = dict(
        cuda = args.cuda,
        seed = args.seed
    )

    env = DmControlEnvForPytorchBothObstype(
        args.domain_name, args.task_name, args.action_repeat)
    dir_name = f'{args.domain_name}-{args.task_name}'

    skill_policy_path = args.policy_path
    log_dir = os.path.join(
        'logs',
        'disentangling',
        dir_name,
        f'slac-seed{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}')

    agent = DisentanglingTrainer(env=env,
                          log_dir=log_dir,
                          skill_policy_path=skill_policy_path,
                          **configs)
    agent.run()

if __name__ == '__main__':
    run()



