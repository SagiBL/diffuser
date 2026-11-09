"""
Import the Hugging Face maze2d Diffuser checkpoint into a jannerm/diffuser clone.

The checkpoints published for RGG (Lee et al., NeurIPS 2023)

    leekwoon/maze2d-large-v1-H384-T256
    leekwoon/maze2d-umaze-v1-H128-T64

are the original Diffuser TemporalUnet weights, converted to a `diffusers`
UNet1DModel + DDPMScheduler. This script downloads one of them, maps the
weights back onto Diffuser's `TemporalUnet`, and writes exactly the files that
`diffuser.utils.load_diffusion` expects:

    logs/<dataset>/diffusion/H<horizon>_T<n_diffusion_steps>/
        dataset_config.pkl  render_config.pkl  model_config.pkl
        diffusion_config.pkl  trainer_config.pkl  state_<epoch>.pt

so that afterwards the stock planner runs unchanged:

    python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1

Usage (from the root of the diffuser clone, `maze2d` branch):

    pip install diffusers huggingface_hub
    python scripts/import_hf_maze2d.py                       # maze2d-large-v1
    python scripts/import_hf_maze2d.py --dataset maze2d-umaze-v1 \
        --hf_repo leekwoon/maze2d-umaze-v1-H128-T64
    python scripts/import_hf_maze2d.py --device cpu          # no GPU needed for the import

Any key of config.maze2d's 'diffusion' block can be overridden the usual way
(e.g. `--logbase logs/pretrained`), and the planner must then be given the same
`--logbase`.
"""
import re
import sys

import torch

import diffuser.utils as utils


# -----------------------------------------------------------------------------#
# ------------------------------- key mapping ---------------------------------#
# -----------------------------------------------------------------------------#

def diffuser_key_to_hf(key):
    """TemporalUnet state_dict key -> diffusers UNet1DModel key.

    This is the inverse of HF's scripts/convert_models_diffuser_to_diffusers.py,
    which zipped the two state_dicts positionally. Written out by name here so
    that a mismatch fails loudly instead of silently permuting weights.
    """
    m = re.fullmatch(r'time_mlp\.(1|3)\.(weight|bias)', key)
    if m:
        return f"time_mlp.{ {'1': 'linear_1', '3': 'linear_2'}[m.group(1)] }.{m.group(2)}"

    m = re.fullmatch(r'final_conv\.0\.block\.(0|2)\.(weight|bias)', key)
    if m:
        return f"out_block.{ {'0': 'final_conv1d_1', '2': 'final_conv1d_gn'}[m.group(1)] }.{m.group(2)}"
    m = re.fullmatch(r'final_conv\.1\.(weight|bias)', key)
    if m:
        return f'out_block.final_conv1d_2.{m.group(1)}'

    m = re.fullmatch(r'downs\.(\d+)\.2\.conv\.(weight|bias)', key)
    if m:
        return f'down_blocks.{m.group(1)}.downsample.conv.{m.group(2)}'
    m = re.fullmatch(r'ups\.(\d+)\.2\.conv\.(weight|bias)', key)
    if m:
        return f'up_blocks.{m.group(1)}.upsample.conv.{m.group(2)}'

    m = re.fullmatch(r'(downs|ups)\.(\d+)\.(0|1)\.(.*)', key)
    if m:
        group = {'downs': 'down_blocks', 'ups': 'up_blocks'}[m.group(1)]
        prefix, rest = f'{group}.{m.group(2)}.resnets.{m.group(3)}', m.group(4)
    else:
        m = re.fullmatch(r'mid_block(1|2)\.(.*)', key)
        if not m:
            raise KeyError(f'unrecognised TemporalUnet key: {key}')
        prefix, rest = f'mid_block.resnets.{int(m.group(1)) - 1}', m.group(2)

    inner = {
        'blocks.0.block.0': 'conv_in.conv1d',
        'blocks.0.block.2': 'conv_in.group_norm',
        'blocks.1.block.0': 'conv_out.conv1d',
        'blocks.1.block.2': 'conv_out.group_norm',
        'time_mlp.1': 'time_emb',
        'residual_conv': 'residual_conv',
    }
    m = re.fullmatch(r'(.*)\.(weight|bias)', rest)
    if not m or m.group(1) not in inner:
        raise KeyError(f'unrecognised TemporalUnet key: {key}')
    return f'{prefix}.{inner[m.group(1)]}.{m.group(2)}'


def hf_to_diffuser_state_dict(hf_state_dict, target_model):
    out, used = {}, set()
    for key, param in target_model.state_dict().items():
        hf_key = diffuser_key_to_hf(key)
        if hf_key not in hf_state_dict:
            raise KeyError(f'{key} -> {hf_key} is missing from the Hugging Face checkpoint')
        tensor = hf_state_dict[hf_key]
        if tuple(tensor.shape) != tuple(param.shape):
            raise ValueError(
                f'shape mismatch for {key}: Diffuser {tuple(param.shape)} vs HF {tuple(tensor.shape)} ({hf_key}). '
                f'Check --dataset / horizon / dim_mults against the HF repo name.')
        out[key] = tensor.clone().to(param.dtype)
        used.add(hf_key)
    unused = set(hf_state_dict) - used
    if unused:
        raise KeyError(f'Hugging Face checkpoint has tensors that were not mapped: {sorted(unused)}')
    return out


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'
    hf_repo: str = 'leekwoon/maze2d-large-v1-H384-T256'
    save_epoch: int = 0          # written as state_<save_epoch>.pt
    skip_dataset: bool = False   # don't build the D4RL dataset just to read obs/action dims


args = Parser().parse_args('diffusion')

try:
    from diffusers import UNet1DModel, DDPMScheduler
except ImportError:
    sys.exit('[ import_hf ] please `pip install diffusers huggingface_hub` into this environment first')


# -----------------------------------------------------------------------------#
# ---------------------- dataset / renderer configs (as train.py) -------------#
# -----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

# -----------------------------------------------------------------------------#
# -------------------------- download the HF checkpoint -----------------------#
# -----------------------------------------------------------------------------#

print(f'[ import_hf ] Downloading {args.hf_repo} from Hugging Face ...')
hf_unet = UNet1DModel.from_pretrained(args.hf_repo, subfolder='unet').eval()
hf_scheduler = DDPMScheduler.from_pretrained(args.hf_repo, subfolder='scheduler')
hf_cfg, sch_cfg = hf_unet.config, hf_scheduler.config

transition_dim = hf_cfg.in_channels

if args.skip_dataset:
    # maze2d: observation = (x, y, vx, vy), action = (ax, ay)
    observation_dim, action_dim = 4, transition_dim - 4
    print(f'[ import_hf ] --skip_dataset: assuming observation_dim={observation_dim}, action_dim={action_dim}')
else:
    dataset = dataset_config()
    observation_dim, action_dim = dataset.observation_dim, dataset.action_dim

assert observation_dim + action_dim == transition_dim, (
    f'HF unet has in_channels={transition_dim} but dataset {args.dataset} has '
    f'observation_dim + action_dim = {observation_dim + action_dim}')

# ---- consistency checks between config.maze2d and the HF checkpoint ----------
hf_dim = hf_cfg.block_out_channels[0]
hf_dim_mults = tuple(c // hf_dim for c in hf_cfg.block_out_channels)
assert hf_dim == 32, f'HF base width {hf_dim} != TemporalUnet default dim=32'
assert hf_dim_mults == tuple(args.dim_mults), \
    f'dim_mults mismatch: config {tuple(args.dim_mults)} vs HF {hf_dim_mults}'
assert sch_cfg.num_train_timesteps == args.n_diffusion_steps, \
    f'n_diffusion_steps mismatch: config {args.n_diffusion_steps} vs HF {sch_cfg.num_train_timesteps}'
assert sch_cfg.beta_schedule == 'squaredcos_cap_v2', f'unexpected beta schedule {sch_cfg.beta_schedule}'
assert (sch_cfg.prediction_type == 'sample') == (not args.predict_epsilon), \
    f'prediction_type {sch_cfg.prediction_type} vs predict_epsilon={args.predict_epsilon}'
assert bool(sch_cfg.clip_sample) == bool(args.clip_denoised), \
    f'clip_sample {sch_cfg.clip_sample} vs clip_denoised={args.clip_denoised}'
m = re.search(r'H(\d+)[-_]T(\d+)', args.hf_repo)
if m and (int(m.group(1)) != args.horizon or int(m.group(2)) != args.n_diffusion_steps):
    print(f'[ import_hf ] WARNING: repo name says H{m.group(1)} T{m.group(2)} but config has '
          f'H{args.horizon} T{args.n_diffusion_steps}; pass --horizon/--n_diffusion_steps to match')

# -----------------------------------------------------------------------------#
# --------------------- model / diffusion / trainer configs -------------------#
# -----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
)

model = model_config()
diffusion = diffusion_config(model)

# -----------------------------------------------------------------------------#
# ------------------------------ transfer weights -----------------------------#
# -----------------------------------------------------------------------------#

state_dict = hf_to_diffuser_state_dict(hf_unet.state_dict(), model)
model.load_state_dict(state_dict, strict=True)
model.eval()
print(f'[ import_hf ] Mapped {len(state_dict)} tensors onto {args.model}')

# the noise schedule must agree too (Diffuser cosine_beta_schedule == squaredcos_cap_v2)
beta_err = (diffusion.betas.cpu() - hf_scheduler.betas.to(diffusion.betas.dtype)).abs().max().item()
assert beta_err < 1e-6, f'beta schedules differ (max |diff| = {beta_err})'

# numerical check: both networks must agree on random inputs
device = next(model.parameters()).device
hf_unet = hf_unet.to(device)
x = torch.randn(2, args.horizon, transition_dim, device=device)
worst = 0.0
with torch.no_grad():
    for t in [0, args.n_diffusion_steps // 2, args.n_diffusion_steps - 1]:
        ts = torch.full((2,), t, dtype=torch.long, device=device)
        y_diffuser = model(x, None, ts)
        y_hf = hf_unet(x.permute(0, 2, 1), ts).sample.permute(0, 2, 1)
        worst = max(worst, (y_diffuser - y_hf).abs().max().item())
assert worst < 1e-4, f'converted network disagrees with HF network (max |diff| = {worst})'
print(f'[ import_hf ] Forward-pass check passed (max |diff| = {worst:.2e})')

# -----------------------------------------------------------------------------#
# ------------------------------------ save -----------------------------------#
# -----------------------------------------------------------------------------#

# Trainer.save() format; EMA == model since we only have one set of weights
data = {'step': 0, 'model': diffusion.state_dict(), 'ema': diffusion.state_dict()}
savepath = f'{args.savepath}/state_{args.save_epoch}.pt'
torch.save(data, savepath)
print(f'[ import_hf ] Saved {savepath}')
print(f'[ import_hf ] Done. Plan with:\n'
      f'    python scripts/plan_maze2d.py --config {args.config} --dataset {args.dataset} '
      f'--logbase {args.logbase}')
