import os
import sys
import csv
import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
from diffuser.guides.obstacles import ObstacleGuide, parse_obstacles
import diffuser.datasets as datasets
import diffuser.utils as utils


# Support flexible CLI argument syntax (e.g. `guidance_method x0`, `--guidance_method=x0`, `--scale 0.2`)
_ALIAS_MAP = {
    'method': 'guidance_method',
    '--method': 'guidance_method',
    'guidance_method': 'guidance_method',
    '--guidance_method': 'guidance_method',
    'scale': 'guidance_scale',
    '--scale': 'guidance_scale',
    'guidance_scale': 'guidance_scale',
    '--guidance_scale': 'guidance_scale',
    'margin': 'obstacle_margin',
    '--margin': 'obstacle_margin',
    'obstacle_margin': 'obstacle_margin',
    '--obstacle_margin': 'obstacle_margin',
    'obstacles': 'obstacles',
    '--obstacles': 'obstacles',
    'obstacle': 'obstacles',
    '--obstacle': 'obstacles',
    'n_guide_steps': 'n_guide_steps',
    '--n_guide_steps': 'n_guide_steps',
    't_stopgrad': 't_stopgrad',
    '--t_stopgrad': 't_stopgrad',
    'use_guidance': 'use_guidance',
    '--use_guidance': 'use_guidance',
    'scale_grad_by_std': 'scale_grad_by_std',
    '--scale_grad_by_std': 'scale_grad_by_std',
    'smoothness_weight': 'smoothness_weight',
    '--smoothness_weight': 'smoothness_weight',
    'use_basin_escape': 'use_basin_escape',
    '--use_basin_escape': 'use_basin_escape',
    'save_name': 'save_name',
    '--save_name': 'save_name',
    'exp_name': 'save_name',
    '--exp_name': 'save_name',
    'folder': 'save_name',
    '--folder': 'save_name',
    'folder_name': 'save_name',
    '--folder_name': 'save_name',
    'seed': 'seed',
    '--seed': 'seed',
}

new_argv = [sys.argv[0]]
i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if '=' in arg:
        k_part, v_part = arg.split('=', 1)
        if k_part in _ALIAS_MAP:
            new_argv.append(f'--{_ALIAS_MAP[k_part]}')
            new_argv.append(v_part)
            i += 1
            continue
    if arg in _ALIAS_MAP:
        new_argv.append(f'--{_ALIAS_MAP[arg]}')
        if i + 1 < len(sys.argv):
            new_argv.append(sys.argv[i + 1])
            i += 2
            continue
    new_argv.append(arg)
    i += 1
sys.argv = new_argv


class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'


#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

#---------------------------------- obstacle guidance setup ----------------------------------#

raw_obstacles = getattr(args, 'obstacles', None)
parsed_obstacles = parse_obstacles(raw_obstacles)
obstacle_margin = getattr(args, 'obstacle_margin', 0.3)
use_guidance = getattr(args, 'use_guidance', True)
guidance_method = getattr(args, 'guidance_method', 'x0')
guidance_scale = getattr(args, 'guidance_scale', 0.15)
n_guide_steps = getattr(args, 'n_guide_steps', 2)
t_stopgrad = getattr(args, 't_stopgrad', 0)
scale_grad_by_std = getattr(args, 'scale_grad_by_std', False)
smoothness_weight = getattr(args, 'smoothness_weight', 0.05)
n_substeps = getattr(args, 'n_substeps', 5)
use_basin_escape = getattr(args, 'use_basin_escape', False)

# Always create the evaluator/guide object if obstacles are defined, and attach to renderer
obstacle_evaluator = ObstacleGuide(
    obstacles=parsed_obstacles,
    normalizer=dataset.normalizer,
    action_dim=dataset.normalizer.action_dim,
    margin=obstacle_margin,
    n_substeps=n_substeps,
    smoothness_weight=smoothness_weight,
).to(diffusion.betas.device)

if hasattr(renderer, 'set_obstacles'):
    renderer.set_obstacles(parsed_obstacles, margin=obstacle_margin)

if use_guidance and len(parsed_obstacles) > 0 and str(guidance_method).lower() not in ('none', 'false', 'off'):
    active_guide = obstacle_evaluator
    diffusion.guide = active_guide
    diffusion.guidance_method = guidance_method
    diffusion.guidance_scale = guidance_scale
    diffusion.n_guide_steps = n_guide_steps
    diffusion.t_stopgrad = t_stopgrad
    diffusion.scale_grad_by_std = scale_grad_by_std
    diffusion.use_basin_escape = use_basin_escape
    print(
        f'[ plan_maze2d ] Gradient Guidance ENABLED | method={guidance_method} | '
        f'scale={guidance_scale} | steps={n_guide_steps} | margin={obstacle_margin} | '
        f'basin_escape={use_basin_escape} | obstacles={parsed_obstacles}'
    )
else:
    active_guide = None
    diffusion.guide = None
    print(
        f'[ plan_maze2d ] Gradient Guidance DISABLED | obstacles={parsed_obstacles}'
    )

policy = Policy(
    diffusion,
    dataset.normalizer,
    guide=active_guide,
    guidance_method=guidance_method,
    guidance_scale=guidance_scale,
    n_guide_steps=n_guide_steps,
    t_stopgrad=t_stopgrad,
    scale_grad_by_std=scale_grad_by_std,
    use_basin_escape=use_basin_escape,
)

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
collision_metrics = {}

total_reward = 0
for t in range(env.max_episode_steps):

    state = env.state_vector().copy()

    ## can replan if desired, but the open-loop plans are good enough for maze2d
    ## that we really only need to plan once
    if t == 0:
        cond[0] = observation
        if getattr(args, 'seed', None) is not None:
            utils.set_seed(int(args.seed))

        action, samples = policy(cond, batch_size=args.batch_size)

        # Evaluate obstacle avoidance metrics across all planned samples
        collision_metrics = obstacle_evaluator.evaluate_trajectories(samples.observations)
        if len(parsed_obstacles) > 0:
            print(
                f'[ plan_maze2d ] Obstacle Avoidance Summary: '
                f'{collision_metrics["n_collision_free"]}/{collision_metrics["n_samples"]} '
                f'({collision_metrics["collision_free_rate"]*100:.1f}%) collision-free | '
                f'min_clearance={collision_metrics["min_clearance"]:.3f} | '
                f'mean_clearance={collision_metrics["mean_min_clearance"]:.3f}'
            )

        # Pick the first collision-free sample (or safest sample) for physical rollout execution
        best_idx = int(np.argmax(collision_metrics['min_sdf_per_sample'])) if len(parsed_obstacles) > 0 else 0
        if collision_metrics['safety_per_sample'][0]:
            exec_idx = 0
        else:
            exec_idx = best_idx

        actions = samples.actions[exec_idx]
        sequence = samples.observations[exec_idx]

        ## save plans as .npy files in waypoints folder (outer folder)
        parent_dir = os.path.dirname(args.savepath) if os.path.basename(args.savepath).isdigit() else args.savepath
        waypoints_dir = join(parent_dir, 'waypoints')
        os.makedirs(waypoints_dir, exist_ok=True)

        for i, sample in enumerate(samples.observations):
            np.save(join(waypoints_dir, f'{i}_waypoints.npy'), sample)

        # combined .npy file of all samples
        np.save(join(waypoints_dir, 'all_planned_waypoints.npy'), samples.observations)

        # Save trajectories.csv for compatibility with plot_tsne.py / classify_tsne.py
        csv_path = join(parent_dir, 'trajectories.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['run_id', 'safety', 'total_violations', 'min_sdf', 'score'])
            writer.writeheader()
            for i in range(len(samples.observations)):
                writer.writerow({
                    'run_id': i,
                    'safety': collision_metrics['safety_per_sample'][i],
                    'total_violations': collision_metrics['violations_per_sample'][i],
                    'min_sdf': round(collision_metrics['min_sdf_per_sample'][i], 4),
                    'score': 0.0,
                })

        print(f'[ plan_maze2d ] Saved {len(samples.observations)} sample(s) to {waypoints_dir}')

    if t < len(sequence) - 1:
        next_waypoint = sequence[t+1]
    else:
        next_waypoint = sequence[-1].copy()
        next_waypoint[2:] = 0

    ## can use actions or define a simple controller based on state predictions
    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

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

    if t % args.vis_freq == 0 or terminal:
        if t == 0:
            renderer.render_samples(args.savepath, samples.observations)

    if terminal:
        break

    observation = next_observation

## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {
    'score': float(score),
    'step': int(t),
    'return': float(total_reward),
    'term': bool(terminal),
    'epoch_diffusion': diffusion_experiment.epoch,
    'use_guidance': bool(use_guidance),
    'guidance_method': str(guidance_method),
    'use_basin_escape': bool(use_basin_escape),
    'guidance_scale': float(guidance_scale),
    'n_guide_steps': int(n_guide_steps),
    'obstacle_margin': float(obstacle_margin),
    'obstacles': parsed_obstacles,
    'collision_free_rate': collision_metrics.get('collision_free_rate', 1.0),
    'n_collision_free': collision_metrics.get('n_collision_free', args.batch_size),
    'min_clearance': collision_metrics.get('min_clearance', float('inf')),
    'mean_min_clearance': collision_metrics.get('mean_min_clearance', float('inf')),
}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
