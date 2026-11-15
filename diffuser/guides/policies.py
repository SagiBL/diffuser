from collections import namedtuple
import numpy as np
import torch
import einops
import pdb

import diffuser.utils as utils

Trajectories = namedtuple('Trajectories', 'actions observations')
GuidedTrajectories = namedtuple('GuidedTrajectories', 'actions observations values')


class Policy:

    def __init__(self, diffusion_model, normalizer, guide=None, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim
        self.guide = guide
        self.sample_kwargs = sample_kwargs

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device=self.device)
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=1, **kwargs):
        conditions = self._format_conditions(conditions, batch_size)

        call_kwargs = dict(self.sample_kwargs)
        call_kwargs.update(kwargs)
        if self.guide is not None and 'guide' not in call_kwargs:
            call_kwargs['guide'] = self.guide

        ## run reverse diffusion process
        sample = self.diffusion_model(conditions, **call_kwargs)

        if isinstance(sample, tuple):
            sample = sample[0]

        sample = utils.to_np(sample)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[0, 0]

        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(actions, observations)
        return action, trajectories


class GuidedPolicy(Policy):
    """
    Convenience subclass of Policy with explicit `guide` argument first,
    matching Diffuser's GuidedPolicy convention.
    """

    def __init__(self, guide, diffusion_model, normalizer, **sample_kwargs):
        super().__init__(diffusion_model, normalizer, guide=guide, **sample_kwargs)
