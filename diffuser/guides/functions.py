import torch

from diffuser.models.helpers import (
    extract,
    apply_conditioning,
)


@torch.no_grad()
def _langevin_basin_escape(model, x, cond, t, guide, scale=0.15):
    """
    Topological Basin-Escape via Guided Langevin Re-noising at intermediate diffusion steps.
    When an unseen obstacle completely blocks a narrow corridor (leaving zero local clearance
    between the obstacle and the maze walls), a trajectory initialized deep in that corridor's
    attraction basin can get wedged against the maze wall by local gradients.

    At intermediate denoising checkpoints, we inspect the predicted clean trajectory x_0_hat.
    For any sample in the batch that still penetrates the obstacle (`min_sdf < 0.05`):
      - We apply a strong broad-horizon repulsive shift to x_0_hat (or seed from a collision-free
        manifold trajectory in the batch if available after strong gradient steps fail), and
      - Re-noise back to timestep t via q_sample(x_0_hat, t, noise=randn) so the remaining
        reverse diffusion steps refine it into an independent, smooth, manifold-valid trajectory.
    """
    x_recon = model.predict_start_from_noise(x, t=t, noise=model.model(x, cond, t))
    if model.clip_denoised:
        x_recon.clamp_(-1.0, 1.0)
    x_recon = apply_conditioning(x_recon, cond, model.action_dim)

    pos_norm = x_recon[:, :, model.action_dim : model.action_dim + 2]
    pos_world = guide.unnormalize_positions(pos_norm)
    dense_pts = guide.dense_segment_points(pos_world)
    sdfs = guide.compute_obstacle_sdfs(dense_pts)  # [B, N_dense, M]
    min_sdf = sdfs.min(dim=-1).values.min(dim=-1).values  # [B]

    threshold = max(0.15, float(getattr(guide, 'margin', 0.3)))
    colliding = min_sdf < threshold
    if not colliding.any():
        return x

    # First, push colliding x_recon strongly using broad temporal smoothing and re-denoise once
    x_pushed = x_recon.clone()
    for _ in range(6):
        _, grad_x0 = guide.gradients(x_pushed, cond, t, smooth_temporally=True)
        x_pushed = x_pushed + (colliding.float()[:, None, None]) * (scale * 2.5) * grad_x0
        if model.clip_denoised:
            x_pushed.clamp_(-1.0, 1.0)
        x_pushed = apply_conditioning(x_pushed, cond, model.action_dim)

    # Re-noise x_pushed to timestep t and check what the U-Net predicts
    # Fork RNG so basin escape does not alter the main reverse-loop RNG stream for other samples/steps
    with torch.random.fork_rng():
        x_candidate = model.q_sample(x_start=x_pushed, t=t, noise=torch.randn_like(x))
        x_candidate = apply_conditioning(x_candidate, cond, model.action_dim)
        x_cand_recon = model.predict_start_from_noise(
            x_candidate, t=t, noise=model.model(x_candidate, cond, t)
        )
        if model.clip_denoised:
            x_cand_recon.clamp_(-1.0, 1.0)

        cand_pos = guide.unnormalize_positions(x_cand_recon[:, :, model.action_dim : model.action_dim + 2])
        cand_sdfs = guide.compute_obstacle_sdfs(guide.dense_segment_points(cand_pos))
        cand_min_sdf = cand_sdfs.min(dim=-1).values.min(dim=-1).values  # [B]

        # Update samples where candidate improved clearance
        improved = colliding & (cand_min_sdf > min_sdf)
        if improved.any():
            x = torch.where(improved[:, None, None], x_candidate, x)
            min_sdf = torch.where(improved, cand_min_sdf, min_sdf)
            x_recon = torch.where(improved[:, None, None], x_cand_recon, x_recon)

        # If any sample is still trapped in the dead-end corridor:
        still_colliding = min_sdf < threshold
        if still_colliding.any():
            safe_indices = torch.where(~still_colliding)[0]
            bad_indices = torch.where(still_colliding)[0]
            if len(safe_indices) > 0:
                for idx_pos, b_idx in enumerate(bad_indices):
                    donor_idx = safe_indices[idx_pos % len(safe_indices)]
                    donor_x0 = x_recon[donor_idx : donor_idx + 1]
                    fresh_xt = model.q_sample(
                        x_start=donor_x0,
                        t=t[b_idx : b_idx + 1],
                        noise=torch.randn_like(donor_x0),
                    )
                    x[b_idx : b_idx + 1] = apply_conditioning(
                        fresh_xt,
                        {k: v[b_idx : b_idx + 1] for k, v in cond.items()},
                        model.action_dim,
                    )
            else:
                # General fallback (e.g. when batch_size == 1): sample K=6 fresh proposals at step t
                # and select the proposal whose predicted x_0_hat has the highest minimum SDF clearance
                K = 6
                for b_idx in bad_indices:
                    sub_cond = {k: v[b_idx : b_idx + 1].repeat(K, 1) for k, v in cond.items()}
                    t_k = t[b_idx : b_idx + 1].repeat(K)
                    proposals = torch.randn((K, x.shape[1], x.shape[2]), device=x.device, dtype=x.dtype)
                    proposals = apply_conditioning(proposals, sub_cond, model.action_dim)
                    prop_x0 = model.predict_start_from_noise(
                        proposals, t=t_k, noise=model.model(proposals, sub_cond, t_k)
                    )
                    if model.clip_denoised:
                        prop_x0.clamp_(-1.0, 1.0)
                    prop_pos = guide.unnormalize_positions(prop_x0[:, :, model.action_dim : model.action_dim + 2])
                    prop_sdfs = guide.compute_obstacle_sdfs(guide.dense_segment_points(prop_pos))
                    prop_min_sdf = prop_sdfs.min(dim=-1).values.min(dim=-1).values  # [K]
                    best_k = torch.argmax(prop_min_sdf)
                    if prop_min_sdf[best_k] > min_sdf[b_idx]:
                        x[b_idx : b_idx + 1] = proposals[best_k : best_k + 1]

    return x


@torch.no_grad()
def n_step_guided_p_sample(
    model,
    x,
    cond,
    t,
    guide,
    scale=0.15,
    t_stopgrad=15,
    n_guide_steps=2,
    scale_grad_by_std=False,
    use_basin_escape=False,
):
    """
    Standard Diffuser direct noisy-state (x_t) gradient guidance (`guidance_method='xt'`).
    Evaluates the guide objective J(x_t) directly on the noisy trajectory x_t for
    `n_guide_steps` additive gradient ascent updates before taking the reverse diffusion step
    through the Temporal U-Net.
    """
    b = x.shape[0]
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)

    # Stop raw x_t gradient perturbations before the final steps so the U-Net projects
    # the trajectory cleanly onto the valid corridor manifold
    effective_stopgrad = max(int(t_stopgrad), 15)

    y = None
    if (t >= effective_stopgrad).any():
        for _ in range(n_guide_steps):
            with torch.enable_grad():
                y, grad = guide.gradients(x, cond, t, smooth_temporally=True)
            if (y == 0).all():
                break

            if scale_grad_by_std:
                grad = model_std * grad

            grad[t < effective_stopgrad] = 0

            x = x + scale * grad
            x = apply_conditioning(x, cond, model.action_dim)

        # Optional: at intermediate checkpoints, escape dead-end corridors via Langevin re-noising
        if use_basin_escape:
            t_val = int(t[0].item())
            if t_val in (150, 100, 60, 25) and hasattr(guide, 'compute_obstacle_sdfs'):
                x = _langevin_basin_escape(model, x, cond, t, guide, scale=scale)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

    return model_mean + nonzero_mask * model_std * noise, y


@torch.no_grad()
def x0_guided_p_sample(
    model,
    x,
    cond,
    t,
    guide,
    scale=0.15,
    t_stopgrad=10,
    n_guide_steps=2,
    scale_grad_by_std=False,
    use_dps_grad=True,
    use_basin_escape=False,
):
    """
    Clean-Prediction / Diffusion Posterior Sampling (DPS) Reconstruction Guidance (`guidance_method='x0'`).

    Instead of evaluating obstacle potentials on noise-corrupted x_t or pushing trajectories
    off the valid maze manifold into walls:
      1. Predicts the clean trajectory manifold estimate x_0_hat = E[x_0 | x_t] via the denoiser.
      2. Evaluates the obstacle objective J(x_0_hat) and backpropagates through the Temporal U-Net
         wrt x_t (`dps_grad = d J(x_0_hat(x_t, t)) / d x_t`). Because the U-Net Jacobian spans
         the tangent space of valid maze trajectories, this shifts colliding trajectories globally
         into alternative collision-free maze corridors without pushing through maze walls.
      3. Optional (`use_basin_escape=True`): Uses Guided Langevin Basin-Escape at intermediate
         denoising checkpoints to escape completely blocked dead-end corridors.
      4. Computes the reverse diffusion posterior q(x_{t-1} | x_t, x_0_hat) using the U-Net's
         clean prediction from the guided x_t.
    """
    b = x.shape[0]
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    effective_stopgrad = max(int(t_stopgrad), 10)
    active_mask = (t >= effective_stopgrad).float().reshape(b, *((1,) * (len(x.shape) - 1)))

    # Early-heavy guidance schedule: stronger push at early/mid steps when corridor choice forms
    tau = (t.float() / float(model.n_timesteps)).reshape(b, *((1,) * (len(x.shape) - 1)))
    time_weight = 0.5 + 1.5 * tau

    y = None

    if (t >= effective_stopgrad).any():
        if use_dps_grad and hasattr(guide, 'dps_gradients'):
            for _ in range(n_guide_steps):
                y, dps_grad, grad_x0, _ = guide.dps_gradients(
                    model, x, cond, t, clip_denoised=model.clip_denoised
                )
                if (y == 0).all():
                    break
                step_scale = scale * time_weight * (model_std if scale_grad_by_std else 1.0)
                # Combine U-Net manifold pullback with broad temporal gradient during early steps
                combined = (1.0 - 0.5 * tau) * dps_grad + (0.5 * tau) * grad_x0
                x = x + active_mask * step_scale * combined
                x = apply_conditioning(x, cond, model.action_dim)
        else:
            # Fast clean-prediction mode (`x0_fast`): evaluate gradient on x_recon and apply to x_t
            for _ in range(n_guide_steps):
                x_recon_tmp = model.predict_start_from_noise(x, t=t, noise=model.model(x, cond, t))
                if model.clip_denoised:
                    x_recon_tmp.clamp_(-1.0, 1.0)
                y, grad_x0 = guide.gradients(x_recon_tmp, cond, t, smooth_temporally=True)
                if (y == 0).all():
                    break
                step_scale = scale * time_weight * (model_std if scale_grad_by_std else 1.0)
                x = x + active_mask * step_scale * grad_x0
                x = apply_conditioning(x, cond, model.action_dim)

        # Optional: at intermediate denoising checkpoints, escape completely blocked corridors via Langevin re-noising
        if use_basin_escape:
            t_val = int(t[0].item())
            if t_val in (150, 100, 60, 25) and hasattr(guide, 'compute_obstacle_sdfs'):
                x = _langevin_basin_escape(model, x, cond, t, guide, scale=scale)

    model_mean, _, posterior_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    posterior_std = torch.exp(0.5 * posterior_log_variance)

    noise = torch.randn_like(x)
    nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

    return model_mean + nonzero_mask * posterior_std * noise, y


@torch.no_grad()
def guided_p_sample(
    model,
    x,
    cond,
    t,
    guide,
    guidance_method='x0',
    scale=0.15,
    t_stopgrad=10,
    n_guide_steps=2,
    scale_grad_by_std=False,
    use_basin_escape=False,
):
    """
    Unified dispatcher for gradient-guided reverse diffusion sampling step.

    Supported `guidance_method` values:
      - `'x0'` / `'dps'` / `'reconstruction'`: Clean-prediction DPS manifold guidance (recommended)
      - `'x0_fast'` / `'clean'`: Fast clean-prediction guidance without U-Net backward pass
      - `'xt'` / `'direct'` / `'standard'`: Direct noisy-state n-step gradient guidance
    """
    method = str(guidance_method).lower().strip()
    if method in ('xt', 'direct', 'standard', 'noisy'):
        return n_step_guided_p_sample(
            model=model,
            x=x,
            cond=cond,
            t=t,
            guide=guide,
            scale=scale,
            t_stopgrad=t_stopgrad,
            n_guide_steps=n_guide_steps,
            scale_grad_by_std=scale_grad_by_std,
            use_basin_escape=use_basin_escape,
        )
    elif method in ('x0_fast', 'clean', 'fast'):
        return x0_guided_p_sample(
            model=model,
            x=x,
            cond=cond,
            t=t,
            guide=guide,
            scale=scale,
            t_stopgrad=t_stopgrad,
            n_guide_steps=n_guide_steps,
            scale_grad_by_std=scale_grad_by_std,
            use_dps_grad=False,
            use_basin_escape=use_basin_escape,
        )
    elif method in ('x0', 'dps', 'reconstruction', 'hybrid'):
        return x0_guided_p_sample(
            model=model,
            x=x,
            cond=cond,
            t=t,
            guide=guide,
            scale=scale,
            t_stopgrad=t_stopgrad,
            n_guide_steps=n_guide_steps,
            scale_grad_by_std=scale_grad_by_std,
            use_dps_grad=True,
            use_basin_escape=use_basin_escape,
        )
    else:
        raise ValueError(
            f"Unknown guidance_method '{guidance_method}'. "
            f"Choose from: 'x0' (or 'dps'), 'x0_fast', 'xt' (or 'direct')."
        )
