import argparse
import os
from datetime import datetime

from env import DmControlEnvForPytorch, GymEnvForPyTorch, DmControlEnvForPytorchBothObstype
from test import DisentanglingTester


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', type=str, default='cheetah')
    parser.add_argument('--task_name', type=str, default='run')
    parser.add_argument('--env_id', type=str, default='HalfCheetah-v2')
    parser.add_argument('--action_repeat', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--log_folder', type=str, default=None)
    parser.add_argument('--run_comment', type=str, default='')
    args = parser.parse_args()

    # Environment
    env = DmControlEnvForPytorchBothObstype(
        args.domain_name, args.task_name, args.action_repeat
    )

    # Directories
    folder_name = f'Skill_test_videos{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}'
    folder_name += args.run_comment

    # Latent model path
    model_path = args.model_path

    # Tester
    tester = DisentanglingTester(
        latent_model_path=model_path,
        env=env,
        seed=args.seed,
        video_dir='test_folder'
    )

    tester.run()


if __name__ == '__main__':
    run()
