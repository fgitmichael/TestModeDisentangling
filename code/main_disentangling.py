import os
import argparse
from datetime import datetime

from env import DmControlEnvForPytorch, GymEnvForPyTorch, DmControlEnvForPytorchBothObstype
from disentangling_trainer import DisentanglingTrainer


def run():
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', type=str, default='cheetah')
    parser.add_argument('--task_name', type=str, default='run')
    parser.add_argument('--env_id', type=str, default='HalfCheetah-v2')
    parser.add_argument('--action_repeat', type=int, default=4)
    parser.add_argument('--cuda', action='store_false')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy_path', type=str,
                        default='./data/01_Skill_policy_half_cheetah/params.pkl')
    parser.add_argument('--log_folder', type=str, default=None)
    parser.add_argument('--run_comment', type=str, default='')
    parser.add_argument('--state_rep', action='store_true')
    args = parser.parse_args()

    # Config dict
    configs = dict(
        cuda=args.cuda,
        seed=args.seed
    )

    # Environment
    if args.state_rep:
        env = DmControlEnvForPytorch(
            args.domain_name,
            args.task_name,
            args.action_repeat,
            obs_type='state'
        )
    else:
        env = DmControlEnvForPytorchBothObstype(
            args.domain_name, args.task_name, args.action_repeat
        )
        feature_dim = 256

    # Directories
    run_id = f'slac-seed{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}'
    dir_name = f'{args.domain_name}-{args.task_name}'
    log_dir_base = os.path.join(
        'logs',
        'disentangling',
        dir_name,
    )
    if args.log_folder is None:
        log_dir = os.path.join(
            log_dir_base,
            run_id
        )
    else:
        log_dir = os.path.join(
            log_dir_base,
            args.log_folder
        )

    # Policy path
    skill_policy_path = args.policy_path

    # Feature dimensions
    if args.state_rep:
        # Set observation dimension as feature_dim
        feature_dim = int(env.observation_space.shape[0])
    else:
        # Use default value of the trainer object
        feature_dim = None

    # Trainer
    agent = DisentanglingTrainer(env=env,
                                 log_dir=log_dir,
                                 skill_policy_path=skill_policy_path,
                                 run_id=run_id,
                                 run_comment=args.run_comment,
                                 state_rep=args.state_rep,
                                 feature_dim=feature_dim,
                                 **configs)
    agent.run()


if __name__ == '__main__':
    run()



