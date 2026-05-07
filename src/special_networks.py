from jaxrl_m.typing import *
from jaxrl_m.networks import *


class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x


class LayerNormRepresentation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = LayerNormMLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final)(observations)


class Representation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = MLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final, activations=nn.gelu)(observations)


class GoalConditionedValue(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    use_layer_norm: bool = True
    ensemble: bool = True
    encoder: nn.Module = None

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        value_net = repr_class((*self.hidden_dims, 1),
                               activate_final=False, ensemble=self.ensemble)
        if self.encoder is not None:
            value_net = nn.Sequential([self.encoder(), value_net])
        self.value_net = value_net

    def __call__(self, observations, goals=None, info=False):
        if goals is None:
            v = self.value_net(observations).squeeze(-1)
        else:
            v = self.value_net(jnp.concatenate(
                [observations, goals], axis=-1)).squeeze(-1)

        return v


class GoalConditionedTopologicalPhiValue(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    skill_dim: int = 2
    use_layer_norm: bool = True
    ensemble: bool = True
    beta: float = 0.3
    encoder: nn.Module = None

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        phi_net = repr_class((*self.hidden_dims, self.skill_dim),
                             activate_final=False, ensemble=self.ensemble)
        if self.encoder is not None:
            phi_net = nn.Sequential([self.encoder(), phi_net])
        self.phi_net = phi_net

    # (singe vf)
    def get_phi(self, observations):
        return self.phi_net(observations)[0]

    def get_all_phis(self, observations):
        return self.phi_net(observations)

    # For bellman target
    def __call__(self, observations, goals=None, task=None, info=False):
        phi_s = self.phi_net(observations)  # (Ens, B, Dim)
        phi_g = self.phi_net(goals)

        diff = phi_g - phi_s
        dist = jnp.sqrt(jnp.maximum(
            jnp.sum(jnp.square(diff), axis=-1, keepdims=True), 1e-6))

        cos_sim = jnp.sum((diff / dist) * task, axis=-1, keepdims=True)

        v = -dist * jnp.exp(self.beta * (1.0 - cos_sim))
        return jnp.squeeze(v, axis=-1)


class TaskModel(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    skill_dim: int = 32
    use_layer_norm: bool = True
    ensemble: bool = True
    encoder: nn.Module = None

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        task_net = repr_class((*self.hidden_dims, self.skill_dim),
                              activate_final=False, ensemble=self.ensemble)
        if self.encoder is not None:
            task_net = nn.Sequential([self.encoder(), task_net])
        self.task_net = task_net

    def get_task(self, goals):
        task = self.task_net(goals)
        normalized = task / \
            (jnp.linalg.norm(task, axis=-1, keepdims=True) + 1e-6)
        return normalized


class GoalConditionedCritic(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    use_layer_norm: bool = True
    ensemble: bool = True
    encoder: nn.Module = None

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        critic_net = repr_class((*self.hidden_dims, 1),
                                activate_final=False, ensemble=self.ensemble)
        if self.encoder is not None:
            critic_net = nn.Sequential([self.encoder(), critic_net])
        self.critic_net = critic_net

    def __call__(self, observations, goals=None, actions=None, info=False):
        if goals is None:
            q = self.critic_net(jnp.concatenate(
                [observations, actions], axis=-1)).squeeze(-1)
        else:
            q = self.critic_net(jnp.concatenate(
                [observations, goals, actions], axis=-1)).squeeze(-1)

        return q


def get_rep(
        encoder: nn.Module, targets: jnp.ndarray, bases: jnp.ndarray = None,
):
    if encoder is None:
        return targets
    else:
        if bases is None:
            return encoder(targets)
        else:
            return encoder(targets, bases)


class IELNetwork(nn.Module):
    networks: Dict[str, nn.Module]

    def unsqueeze_context(self, observations, contexts):
        if len(observations.shape) <= 2:
            return contexts
        else:
            # observations: (H, W, D) or (B, H, W, D)
            # contexts: (Z) -> (H, W, Z) or (B, Z) -> (B, H, W, Z)
            assert len(observations.shape) == len(contexts.shape) + 2
            return jnp.expand_dims(jnp.expand_dims(contexts, axis=-2), axis=-2).repeat(observations.shape[-3], axis=-3).repeat(observations.shape[-2], axis=-2)

    def value(self, observations, goals=None, task=None, **kwargs):
        return self.networks['value'](observations, goals, task, **kwargs)

    def target_value(self, observations, goals=None, task=None, **kwargs):
        return self.networks['target_value'](observations, goals, task, **kwargs)

    def phi(self, observations, **kwargs):
        return self.networks['value'].get_phi(observations, **kwargs)

    def get_all_phis(self, observations, **kwargs):
        return self.networks['value'].get_all_phis(observations, **kwargs)

    def get_task(self, goals, **kwargs):
        return self.networks['task'].get_task(goals, **kwargs)

    def skill_value(self, observations, skills, **kwargs):
        skills = self.unsqueeze_context(observations, skills)
        return self.networks['skill_value'](observations, skills, **kwargs)

    def skill_target_value(self, observations, skills, **kwargs):
        skills = self.unsqueeze_context(observations, skills)
        return self.networks['skill_target_value'](observations, skills, **kwargs)

    def skill_critic(self, observations, skills, actions=None, **kwargs):
        skills = self.unsqueeze_context(observations, skills)
        actions = self.unsqueeze_context(observations, actions)
        return self.networks['skill_critic'](observations, skills, actions, **kwargs)

    def skill_target_critic(self, observations, skills, actions=None, **kwargs):
        skills = self.unsqueeze_context(observations, skills)
        actions = self.unsqueeze_context(observations, actions)
        return self.networks['skill_target_critic'](observations, skills, actions, **kwargs)

    def skill_actor(self, observations, skills, **kwargs):
        skills = self.unsqueeze_context(observations, skills)
        return self.networks['skill_actor'](jnp.concatenate([observations, skills], axis=-1), **kwargs)

    def __call__(self, observations, goals, actions, skills):
        # Only for initialization
        rets = {
            'value': self.value(observations, goals, skills),
            'target_value': self.target_value(observations, goals, skills),
            'task': self.get_task(goals),
            'skill_actor': self.skill_actor(observations, skills),
            'skill_value': self.skill_value(observations, skills),
            'skill_critic': self.skill_critic(observations, skills, actions),
            'skill_target_critic': self.skill_target_critic(observations, skills, actions),
        }
        return rets
