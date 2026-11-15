import ast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_obstacles(obstacles_spec):
    """
    Parse flexible obstacle specifications into a list of canonical obstacle dicts.

    Supported formats in world coordinates (y, x):
      - Circular obstacle:
          (y, x, r) or [y, x, r]
          {'type': 'circle', 'y': y, 'x': x, 'r': r}
          {'type': 'circle', 'center': (y, x), 'radius': r}
      - Axis-aligned box / grid obstacle:
          (ymin, ymax, xmin, xmax) or [ymin, ymax, xmin, xmax]
          {'type': 'box', 'bounds': (ymin, ymax, xmin, xmax)}
          {'type': 'box', 'ymin': ymin, 'ymax': ymax, 'xmin': xmin, 'xmax': xmax}
          {'type': 'box', 'center': (cy, cx), 'size': (height, width)}
      - String representation of any of the above, or semicolon-separated shorthand:
          "circle:3.0,6.0,0.6;box:2.4,3.6,5.4,6.6"
    """
    if obstacles_spec is None:
        return []

    if isinstance(obstacles_spec, str):
        spec_str = obstacles_spec.strip()
        if not spec_str or spec_str.lower() in ('none', 'null', 'false', '[]'):
            return []
        if ':' in spec_str and not spec_str.startswith(('[', '{', '(')):
            items = []
            for part in spec_str.split(';'):
                part = part.strip()
                if not part:
                    continue
                if ':' in part:
                    kind, vals_str = part.split(':', 1)
                    vals = [float(v.strip()) for v in vals_str.split(',')]
                    kind = kind.strip().lower()
                    if kind in ('circle', 'sphere', 'disk', 'c'):
                        if len(vals) != 3:
                            raise ValueError(f"Circle obstacle requires 3 values (y, x, r), got: {part}")
                        items.append({'type': 'circle', 'center': (vals[0], vals[1]), 'radius': vals[2]})
                    elif kind in ('box', 'rect', 'grid', 'b'):
                        if len(vals) != 4:
                            raise ValueError(f"Box obstacle requires 4 values (ymin, ymax, xmin, xmax), got: {part}")
                        items.append({'type': 'box', 'bounds': (vals[0], vals[1], vals[2], vals[3])})
                    else:
                        raise ValueError(f"Unknown obstacle type '{kind}' in '{part}'")
                else:
                    vals = [float(v.strip()) for v in part.split(',')]
                    items.append(tuple(vals))
            return parse_obstacles(items)

        try:
            obstacles_spec = ast.literal_eval(spec_str)
        except Exception as e:
            raise ValueError(f"Could not parse obstacles string '{obstacles_spec}': {e}")

    if isinstance(obstacles_spec, (tuple, list)) and len(obstacles_spec) in (3, 4) and all(
        isinstance(v, (int, float, np.number)) for v in obstacles_spec
    ):
        obstacles_spec = [obstacles_spec]
    elif isinstance(obstacles_spec, dict):
        obstacles_spec = [obstacles_spec]

    parsed = []
    for item in obstacles_spec:
        if isinstance(item, (tuple, list)):
            if len(item) == 3:
                y, x, r = float(item[0]), float(item[1]), float(item[2])
                parsed.append({
                    'type': 'circle',
                    'center': (y, x),
                    'radius': r,
                })
            elif len(item) == 4:
                ymin, ymax, xmin, xmax = float(item[0]), float(item[1]), float(item[2]), float(item[3])
                if ymin > ymax:
                    ymin, ymax = ymax, ymin
                if xmin > xmax:
                    xmin, xmax = xmax, xmin
                parsed.append({
                    'type': 'box',
                    'bounds': (ymin, ymax, xmin, xmax),
                })
            else:
                raise ValueError(
                    f"Tuple/list obstacle must have length 3 (y, x, r) or 4 (ymin, ymax, xmin, xmax), got: {item}"
                )
        elif isinstance(item, dict):
            obs_type = str(item.get('type', '')).lower()
            if obs_type in ('circle', 'sphere', 'disk') or ('radius' in item or 'r' in item):
                if 'center' in item:
                    cy, cx = float(item['center'][0]), float(item['center'][1])
                else:
                    cy, cx = float(item['y']), float(item['x'])
                r = float(item.get('radius', item.get('r')))
                parsed.append({
                    'type': 'circle',
                    'center': (cy, cx),
                    'radius': r,
                })
            elif obs_type in ('box', 'rect', 'rectangle', 'grid') or 'bounds' in item or 'ymin' in item:
                if 'bounds' in item:
                    ymin, ymax, xmin, xmax = [float(v) for v in item['bounds']]
                elif 'center' in item and 'size' in item:
                    cy, cx = float(item['center'][0]), float(item['center'][1])
                    h, w = float(item['size'][0]), float(item['size'][1])
                    ymin, ymax = cy - 0.5 * h, cy + 0.5 * h
                    xmin, xmax = cx - 0.5 * w, cx + 0.5 * w
                else:
                    ymin, ymax = float(item['ymin']), float(item['ymax'])
                    xmin, xmax = float(item['xmin']), float(item['xmax'])
                if ymin > ymax:
                    ymin, ymax = ymax, ymin
                if xmin > xmax:
                    xmin, xmax = xmax, xmin
                parsed.append({
                    'type': 'box',
                    'bounds': (ymin, ymax, xmin, xmax),
                })
            else:
                raise ValueError(f"Unrecognized obstacle dictionary format: {item}")
        else:
            raise ValueError(f"Unrecognized obstacle specification item: {item}")

    return parsed


class ObstacleGuide(nn.Module):
    """
    Differentiable Signed Distance Field (SDF) Obstacle Avoidance Guide for Diffuser.

    Evaluates a smooth repulsive potential over dense linearly-interpolated segments
    between consecutive trajectory waypoints (preventing waypoint tunneling through
    obstacles) and applies CHOMP-style multi-scale Gaussian temporal smoothing to the
    spatial obstacle gradients so trajectories bend smoothly around obstacles without
    high-frequency oscillations.

    Crucially, if a trajectory does not enter any obstacle's safety margin (`sdf >= margin`),
    both the cost and gradient are identically zero, leaving non-colliding trajectories untouched.
    """

    def __init__(
        self,
        obstacles,
        normalizer,
        action_dim=2,
        margin=0.3,
        n_substeps=5,
        smoothness_weight=0.05,
        linear_penalty=1.5,
        max_grad_norm=0.5,
    ):
        super().__init__()
        self.obstacles = parse_obstacles(obstacles)
        self.normalizer = normalizer
        self.action_dim = action_dim
        self.margin = float(margin)
        self.n_substeps = max(1, int(n_substeps))
        self.smoothness_weight = float(smoothness_weight)
        self.linear_penalty = float(linear_penalty)
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None

        obs_norm = normalizer.normalizers['observations']
        obs_mins = torch.tensor(obs_norm.mins[:2], dtype=torch.float32)
        obs_maxs = torch.tensor(obs_norm.maxs[:2], dtype=torch.float32)
        self.register_buffer('obs_mins', obs_mins)
        self.register_buffer('obs_maxs', obs_maxs)

        circles = [o for o in self.obstacles if o['type'] == 'circle']
        boxes = [o for o in self.obstacles if o['type'] == 'box']

        if circles:
            c_centers = torch.tensor([o['center'] for o in circles], dtype=torch.float32)
            c_radii = torch.tensor([o['radius'] for o in circles], dtype=torch.float32)
        else:
            c_centers = torch.zeros((0, 2), dtype=torch.float32)
            c_radii = torch.zeros((0,), dtype=torch.float32)

        if boxes:
            b_centers = torch.tensor(
                [[0.5 * (o['bounds'][0] + o['bounds'][1]), 0.5 * (o['bounds'][2] + o['bounds'][3])] for o in boxes],
                dtype=torch.float32,
            )
            b_halfs = torch.tensor(
                [[0.5 * (o['bounds'][1] - o['bounds'][0]), 0.5 * (o['bounds'][3] - o['bounds'][2])] for o in boxes],
                dtype=torch.float32,
            )
        else:
            b_centers = torch.zeros((0, 2), dtype=torch.float32)
            b_halfs = torch.zeros((0, 2), dtype=torch.float32)

        self.register_buffer('circle_centers', c_centers)
        self.register_buffer('circle_radii', c_radii)
        self.register_buffer('box_centers', b_centers)
        self.register_buffer('box_halfs', b_halfs)

        # Build multi-scale 1D Gaussian smoothing kernels for CHOMP-style temporal gradient smoothing
        self.register_buffer('kernel_local', self._make_gaussian_kernel(kernel_size=25, sigma=5.0))
        self.register_buffer('kernel_broad', self._make_gaussian_kernel(kernel_size=65, sigma=16.0))

    @staticmethod
    def _make_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        # Normalize so peak is 1.0 (preserves displacement magnitude at obstacle center while spreading it temporally)
        g = g / g.max()
        return g.view(1, 1, kernel_size)

    def _smooth_temporal_grad(self, pos_grad: torch.Tensor) -> torch.Tensor:
        """
        Smooth spatial gradients along the trajectory horizon using a multi-scale
        1D Gaussian filter so local obstacle pushes deform the trajectory smoothly
        without kinks or high-frequency oscillations.
        pos_grad: [B, H, 2]
        """
        B, H, D = pos_grad.shape
        if H < 5:
            return pos_grad

        # Reshape to [B * D, 1, H] for 1D convolution along horizon H
        g_in = pos_grad.permute(0, 2, 1).reshape(B * D, 1, H)

        k_local = self.kernel_local.to(pos_grad.device, dtype=pos_grad.dtype)
        k_broad = self.kernel_broad.to(pos_grad.device, dtype=pos_grad.dtype)

        pad_l = k_local.shape[-1] // 2
        pad_b = k_broad.shape[-1] // 2

        # Normalize convolution by sum of weights so magnitude stays well-calibrated
        g_local = F.conv1d(g_in, k_local / k_local.sum(), padding=pad_l) * 3.0
        g_broad = F.conv1d(g_in, k_broad / k_broad.sum(), padding=pad_b) * 5.0

        g_smooth = 0.5 * g_local + 0.5 * g_broad
        g_out = g_smooth.reshape(B, D, H).permute(0, 2, 1)

        # Taper at fixed start and goal endpoints so boundary conditions are never fighting the push
        ramp_len = min(12, H // 4)
        if ramp_len > 0:
            taper = torch.ones(H, device=pos_grad.device, dtype=pos_grad.dtype)
            ramp = torch.linspace(0.0, 1.0, ramp_len, device=pos_grad.device, dtype=pos_grad.dtype)
            taper[:ramp_len] = ramp
            taper[-ramp_len:] = ramp.flip(0)
            g_out = g_out * taper[None, :, None]

        return g_out

    def unnormalize_positions(self, pos_norm):
        """
        Differentiably map normalized positions in [-1, 1] to world coordinates (y, x).
        pos_norm: [..., 2]
        """
        mins = self.obs_mins.to(pos_norm.device)
        maxs = self.obs_maxs.to(pos_norm.device)
        return 0.5 * (pos_norm + 1.0) * (maxs - mins) + mins

    def normalize_positions(self, pos_world):
        """
        Differentiably map world coordinates (y, x) to normalized [-1, 1].
        pos_world: [..., 2]
        """
        mins = self.obs_mins.to(pos_world.device)
        maxs = self.obs_maxs.to(pos_world.device)
        return 2.0 * (pos_world - mins) / (maxs - mins + 1e-8) - 1.0

    def dense_segment_points(self, pos_world):
        """
        Interpolate `n_substeps` points along each segment [p_t, p_{t+1}] to prevent
        trajectories from jumping/tunneling across obstacles between discrete steps.

        pos_world: [B, H, 2]
        Returns: [B, (H - 1) * n_substeps + 1, 2]
        """
        if self.n_substeps <= 1 or pos_world.shape[1] <= 1:
            return pos_world

        p0 = pos_world[:, :-1, :]  # [B, H-1, 2]
        p1 = pos_world[:, 1:, :]   # [B, H-1, 2]
        alphas = torch.linspace(
            0.0, 1.0, self.n_substeps + 1, device=pos_world.device, dtype=pos_world.dtype
        )[:-1]  # [K]
        interp = (1.0 - alphas[None, None, :, None]) * p0[:, :, None, :] + alphas[None, None, :, None] * p1[:, :, None, :]
        B, H_minus_1, K, _ = interp.shape
        dense = interp.reshape(B, H_minus_1 * K, 2)
        return torch.cat([dense, pos_world[:, -1:, :]], dim=1)

    def compute_obstacle_sdfs(self, points):
        """
        Compute Signed Distance Function (SDF) values from `points` to all obstacles.

        points: [B, N, 2] in world coordinates (y, x)
        Returns: sdf tensor of shape [B, N, M] where M is total number of obstacles.
                 sdf < 0 inside an obstacle, sdf == 0 on boundary, sdf > 0 outside.
        """
        sdfs = []
        device = points.device

        if self.circle_centers.shape[0] > 0:
            centers = self.circle_centers.to(device)  # [Mc, 2]
            radii = self.circle_radii.to(device)      # [Mc]
            diff = points[:, :, None, :] - centers[None, None, :, :]  # [B, N, Mc, 2]
            # Small deterministic offset avoids zero-gradient singularity if a point hits exact center
            dist = torch.sqrt(((diff + 1e-4) ** 2).sum(dim=-1) + 1e-6)  # [B, N, Mc]
            sdfs.append(dist - radii[None, None, :])

        if self.box_centers.shape[0] > 0:
            b_centers = self.box_centers.to(device)  # [Mb, 2]
            b_halfs = self.box_halfs.to(device)      # [Mb, 2]
            rel = points[:, :, None, :] - b_centers[None, None, :, :]  # [B, N, Mb, 2]
            q = torch.abs(rel) - b_halfs[None, None, :, :]             # [B, N, Mb, 2]
            outside_dist = torch.sqrt((torch.clamp(q, min=0.0) ** 2).sum(dim=-1) + 1e-6)
            inside_dist = torch.clamp(torch.max(q[..., 0], q[..., 1]), max=0.0)
            normalized_rel = (rel + 1e-4) / (b_halfs[None, None, :, :] + 1e-6)
            radial_dist = torch.sqrt((normalized_rel ** 2).sum(dim=-1) + 1e-6)
            interior_smooth = torch.where(
                inside_dist < 0.0,
                0.25 * (radial_dist - 1.0) * torch.min(b_halfs[:, 0], b_halfs[:, 1])[None, None, :],
                torch.zeros_like(inside_dist),
            )
            sdfs.append(outside_dist + 0.75 * inside_dist + interior_smooth)

        if not sdfs:
            return torch.full((*points.shape[:2], 1), 1e6, device=device, dtype=points.dtype)

        return torch.cat(sdfs, dim=-1)

    def obstacle_cost_from_positions(self, pos_world):
        """
        Compute repulsive obstacle cost per trajectory.
        Returns strictly 0.0 for trajectories that do not enter any obstacle's safety margin.
        pos_world: [B, H, 2] in world coordinates (y, x)
        Returns: cost tensor of shape [B]
        """
        if len(self.obstacles) == 0:
            return torch.zeros(pos_world.shape[0], device=pos_world.device, dtype=pos_world.dtype)

        dense_pts = self.dense_segment_points(pos_world)  # [B, N_dense, 2]
        sdfs = self.compute_obstacle_sdfs(dense_pts)      # [B, N_dense, M]

        # Clearance violation: positive only when sdf < margin, strictly 0 outside margin
        violation = torch.clamp(self.margin - sdfs, min=0.0)  # [B, N_dense, M]
        # Core penetration: positive only when strictly inside obstacle (sdf < 0)
        penetration = torch.clamp(-sdfs, min=0.0)             # [B, N_dense, M]

        point_cost = 0.5 * (violation ** 2) + self.linear_penalty * violation + 2.0 * penetration
        obs_cost = point_cost.sum(dim=(1, 2)) / float(self.n_substeps)
        return obs_cost

    def forward(self, x, cond=None, t=None):
        """
        Compute scalar guidance objective J(x) = -cost(x) for each trajectory in batch.
        x: [B, H, transition_dim] in normalized space [-1, 1]
        Returns: [B]
        """
        pos_norm = x[:, :, self.action_dim : self.action_dim + 2]
        pos_world = self.unnormalize_positions(pos_norm)
        cost = self.obstacle_cost_from_positions(pos_world)
        return -cost

    def _postprocess_grad(self, grad, smooth_temporally=True):
        """
        Restrict gradient to position channels, optionally apply CHOMP-style Gaussian
        temporal smoothing along the horizon, and clip extreme waypoint gradient norms.
        """
        masked = torch.zeros_like(grad)
        pos_grad = grad[:, :, self.action_dim : self.action_dim + 2]

        if smooth_temporally:
            pos_grad = self._smooth_temporal_grad(pos_grad)

        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            pt_norm = torch.norm(pos_grad, dim=-1, keepdim=True) + 1e-8
            scale = torch.clamp(self.max_grad_norm / pt_norm, max=1.0)
            pos_grad = pos_grad * scale

        masked[:, :, self.action_dim : self.action_dim + 2] = pos_grad
        return masked

    def gradients(self, x, cond=None, t=None, smooth_temporally=True):
        """
        Compute objective J(x) and its temporally-smoothed gradient wrt x: (J(x), dJ(x)/dx).
        """
        if len(self.obstacles) == 0:
            return torch.zeros(x.shape[0], device=x.device), torch.zeros_like(x)

        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            y = self.forward(x_in, cond, t)
            if (y == 0).all():
                return y.detach(), torch.zeros_like(x)
            grad = torch.autograd.grad(y.sum(), x_in)[0]

        grad = self._postprocess_grad(grad.detach(), smooth_temporally=smooth_temporally)
        return y.detach(), grad

    def dps_gradients(self, diffusion_model, x_t, cond, t, clip_denoised=True):
        """
        Diffusion Posterior Sampling (DPS) / Clean-Prediction Reconstruction Guidance:
        Predicts the clean trajectory x_0_hat(x_t, t) using the diffusion model,
        evaluates the obstacle objective J(x_0_hat), and backpropagates through the
        Temporal U-Net denoiser wrt x_t:
            grad_xt = d J(x_0_hat(x_t, t)) / d x_t
        For trajectories in the batch that already have 0 obstacle violation (y == 0),
        the gradient is strictly 0 so collision-free trajectories remain completely unperturbed.
        """
        if len(self.obstacles) == 0:
            with torch.no_grad():
                noise_pred = diffusion_model.model(x_t, cond, t)
                x_recon = diffusion_model.predict_start_from_noise(x_t, t=t, noise=noise_pred)
                if clip_denoised:
                    x_recon = x_recon.clamp(-1.0, 1.0)
            zeros = torch.zeros_like(x_t)
            return torch.zeros(x_t.shape[0], device=x_t.device), zeros, zeros, x_recon

        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            noise_pred = diffusion_model.model(x_in, cond, t)
            x_recon = diffusion_model.predict_start_from_noise(x_in, t=t, noise=noise_pred)
            if clip_denoised:
                x_recon = torch.clamp(x_recon, -1.0, 1.0)

            y = self.forward(x_recon, cond, t)
            if (y == 0).all():
                zeros = torch.zeros_like(x_t)
                return y.detach(), zeros, zeros, x_recon.detach()

            grad_xt, grad_x0 = torch.autograd.grad(y.sum(), [x_in, x_recon], retain_graph=False)

        grad_x0_smooth = self._postprocess_grad(grad_x0.detach(), smooth_temporally=True)

        # Normalize the U-Net Jacobian pullback gradient per colliding trajectory
        active_sample = (y < -1e-7).float()[:, None, None]
        xt_max = grad_xt.abs().amax(dim=(1, 2), keepdim=True) + 1e-8
        x0_max = grad_x0_smooth.abs().amax(dim=(1, 2), keepdim=True) + 1e-8
        grad_xt_scaled = grad_xt.detach() * (x0_max / xt_max) * active_sample
        grad_x0_smooth = grad_x0_smooth * active_sample

        return y.detach(), grad_xt_scaled, grad_x0_smooth, x_recon.detach()

    @torch.no_grad()
    def evaluate_trajectories(self, observations_np):
        """
        Evaluate collision statistics on unnormalized observation trajectories.
        observations_np: np.ndarray of shape [B, H, obs_dim]
        Returns dict with per-sample and aggregate metrics.
        """
        pos = torch.tensor(observations_np[:, :, :2], dtype=torch.float32, device=self.obs_mins.device)
        if len(self.obstacles) == 0:
            B = pos.shape[0]
            return {
                'n_samples': int(B),
                'n_collision_free': int(B),
                'collision_free_rate': 1.0,
                'min_sdf_per_sample': [float('inf')] * B,
                'violations_per_sample': [0] * B,
                'safety_per_sample': [True] * B,
                'min_clearance': float('inf'),
                'mean_min_clearance': float('inf'),
            }

        dense_pts = self.dense_segment_points(pos)
        sdfs = self.compute_obstacle_sdfs(dense_pts)  # [B, N_dense, M]
        min_sdf_per_point = sdfs.min(dim=-1).values   # [B, N_dense]
        min_sdf_per_sample = min_sdf_per_point.min(dim=-1).values  # [B]

        wp_sdfs = self.compute_obstacle_sdfs(pos).min(dim=-1).values  # [B, H]
        violations_per_sample = (wp_sdfs < 0.0).sum(dim=-1)           # [B]
        safety_per_sample = (min_sdf_per_sample >= 0.0)

        min_sdf_list = [float(v) for v in min_sdf_per_sample.cpu().numpy()]
        viol_list = [int(v) for v in violations_per_sample.cpu().numpy()]
        safe_list = [bool(v) for v in safety_per_sample.cpu().numpy()]

        return {
            'n_samples': int(pos.shape[0]),
            'n_collision_free': int(sum(safe_list)),
            'collision_free_rate': float(np.mean(safe_list)),
            'min_sdf_per_sample': min_sdf_list,
            'violations_per_sample': viol_list,
            'safety_per_sample': safe_list,
            'min_clearance': float(np.min(min_sdf_list)),
            'mean_min_clearance': float(np.mean(min_sdf_list)),
        }
