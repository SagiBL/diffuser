import os
import json
import numpy as np
from os.path import join
import pdb

import sys

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

# Support flexible temp_high and temp_low argument syntax:
# e.g.:
#   python scripts/plan_maze2d.py temp_high 2.5 temp_low 0.1
#   python scripts/plan_maze2d.py tmp_high 2.5 tmp_low 0.1
#   python scripts/plan_maze2d.py --temp_high 2.5 --temp_low 0.1
new_argv = [sys.argv[0]]
i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg in ('temp_high', 'tmp_high', '--tmp_high'):
        new_argv.append('--temp_high')
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
            new_argv.append(sys.argv[i + 1])
            i += 1
    elif arg.startswith(('temp_high=', 'tmp_high=', '--tmp_high=')):
        new_argv.append('--temp_high')
        new_argv.append(arg.split('=', 1)[1])
    elif arg in ('temp_low', 'tmp_low', '--tmp_low'):
        new_argv.append('--temp_low')
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
            new_argv.append(sys.argv[i + 1])
            i += 1
    elif arg.startswith(('temp_low=', 'tmp_low=', '--tmp_low=')):
        new_argv.append('--temp_low')
        new_argv.append(arg.split('=', 1)[1])
    else:
        new_argv.append(arg)
    i += 1
sys.argv = new_argv


class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'
    temp_high: float = 2.0
    temp_low: float = 0.0

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

diffusion.temp_high = args.temp_high
diffusion.temp_low = args.temp_low
print(f'[ plan_maze2d ] Temperature schedule: temp_high = {diffusion.temp_high}, temp_low = {diffusion.temp_low}')

policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- main loop ----------------------------------#

observation = env.reset()

## set start location if specified in config or runtime args
if hasattr(args, 'start') and args.start is not None:
    start = args.start
    if isinstance(start, str):
        start = eval(start)
    start_pos = np.array(start[:2])
    start_vel = np.array(start[2:]) if len(start) > 2 else np.zeros(2)
    start_state = np.concatenate([start_pos, start_vel])
    utils.set_state(env, start_state)
    observation = env.state_vector().copy()

if args.conditional:
    print('Resetting target')
    env.set_target()

## set conditioning xy position to be the goal
target = env._target
cond = {
    diffusion.horizon - 1: np.array([*target, 0, 0]),
}

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
for t in range(env.max_episode_steps):

    state = env.state_vector().copy()

    ## can replan if desired, but the open-loop plans are good enough for maze2d
    ## that we really only need to plan once
    if t == 0:
        cond[0] = observation

        action, samples = policy(cond, batch_size=args.batch_size)
        actions = samples.actions[0]
        sequence = samples.observations[0]

        ## save plans as .npy files in waypoints folder (outer folder)
        parent_dir = os.path.dirname(args.savepath) if os.path.basename(args.savepath).isdigit() else args.savepath
        waypoints_dir = join(parent_dir, 'waypoints')
        os.makedirs(waypoints_dir, exist_ok=True)

        for i, sample in enumerate(samples.observations):
            np.save(join(waypoints_dir, f'{i}_waypoints.npy'), sample)

        # combined .npy file of all samples
        np.save(join(waypoints_dir, 'all_planned_waypoints.npy'), samples.observations)

        print(f'[ plan_maze2d ] Saved {len(samples.observations)} sample(s) to {waypoints_dir}')
    # pdb.set_trace()

    # ####
    if t < len(sequence) - 1:
        next_waypoint = sequence[t+1]
    else:
        next_waypoint = sequence[-1].copy()
        next_waypoint[2:] = 0
        # pdb.set_trace()

    ## can use actions or define a simple controller based on state predictions
    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
    # pdb.set_trace()
    ####

    # else:
    #     actions = actions[1:]
    #     if len(actions) > 1:
    #         action = actions[0]
    #     else:
    #         # action = np.zeros(2)
    #         action = -state[2:]
    #         pdb.set_trace()



    next_observation, reward, terminal, _ = env.step(action)
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'{action}'
    )

    if 'maze2d' in args.dataset:
        xy = next_observation[:2]
        goal = env.unwrapped._target
        print(
            f'maze | pos: {xy} | goal: {goal}'
        )

    ## update rollout observations
    rollout.append(next_observation.copy())

    # logger.log(score=score, step=t)

    if t % args.vis_freq == 0 or terminal:
        if t == 0: renderer.render_samples(args.savepath, samples.observations)


        # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        ## save rollout thus far
        # renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)

        # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

    if terminal:
        break

    observation = next_observation

# logger.finish(t, env.max_episode_steps, score=score, value=0)

## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
