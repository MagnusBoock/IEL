import copy

from jaxrl_m.typing import *

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState
from jaxrl_m.networks import Policy
from jaxrl_m.vision import encoders

import flax
from flax.core import freeze, unfreeze
from src.special_networks import GoalConditionedValue, GoalConditionedCritic, GoalConditionedTopologicalPhiValue, IELNetwork, TaskModel

all_info_dict = {
    "task_encoder/loss": 0.0,
    "value/value_loss": 0.0,
    "directed_step_loss_state": 0.0,
    "directed_step_loss_next_state": 0.0,
    "skill_value/value_loss": 0.0,
    "skill_critic/critic_loss": 0.0,
    "skill_actor/actor_loss": 0.0,
    "skill_actor/mse": 0.0,
}


def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)


def compute_value_loss(agent, batch, network_params, task):
    # masks are 0 if terminal, 1 otherwise
    batch['masks'] = 1.0 - batch['rewards']
    # rewards are 0 if terminal, -1 otherwise
    batch['rewards'] = batch['rewards'] - 1.0

    (next_v1, next_v2) = agent.network(
        batch['next_observations'], batch['goals'], task, method='target_value')
    next_v = jnp.minimum(next_v1, next_v2)
    q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v

    (v1_t, v2_t) = agent.network(
        batch['observations'], batch['goals'], task, method='target_value')
    v_t = (v1_t + v2_t) / 2
    adv = q - v_t

    q1 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v1
    q2 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v2
    (v1, v2) = agent.network(
        batch['observations'], batch['goals'], task, method='value', params=network_params)
    v = (v1 + v2) / 2

    value_loss1 = expectile_loss(
        adv, q1 - v1, agent.config['expectile']).mean()
    value_loss2 = expectile_loss(
        adv, q2 - v2, agent.config['expectile']).mean()
    value_loss = value_loss1 + value_loss2

    return value_loss, {
        'value_loss': value_loss,
    }


def compute_skill_value_loss(agent, batch, network_params):
    q1, q2 = agent.network(batch['observations'], batch['skills'],
                           batch['actions'], method='skill_target_critic')
    q = jnp.minimum(q1, q2)
    v = agent.network(batch['observations'], batch['skills'],
                      method='skill_value', params=network_params)
    adv = q - v
    value_loss = expectile_loss(
        adv, q - v, agent.config['skill_expectile']).mean()

    return value_loss, {
        'value_loss': value_loss,
    }


def compute_skill_critic_loss(agent, batch, network_params):
    next_v = agent.network(
        batch['next_observations'], batch['skills'], method='skill_value')
    q = batch['rewards'] + agent.config['skill_discount'] * next_v  # No 'done'

    q1, q2 = agent.network(batch['observations'], batch['skills'],
                           batch['actions'], method='skill_critic', params=network_params)
    critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

    return critic_loss, {
        'critic_loss': critic_loss,
    }


def compute_skill_actor_loss(agent, batch, network_params):
    v = agent.network(batch['observations'],
                      batch['skills'], method='skill_value')
    q1, q2 = agent.network(batch['observations'], batch['skills'],
                           batch['actions'], method='skill_target_critic')
    q = jnp.minimum(q1, q2)
    adv = q - v

    exp_a = jnp.exp(adv * agent.config['skill_temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)

    dist = agent.network(batch['observations'], batch['skills'],
                         method='skill_actor', params=network_params)
    log_probs = dist.log_prob(batch['actions'])
    actor_loss = -(exp_a * log_probs).mean()

    return actor_loss, {
        'actor_loss': actor_loss,
        'mse': jnp.mean((dist.mode() - batch['actions'])**2),
    }


def compute_continuous_contrastive_loss(agent, batch, network_params, temperature=0.07):
    g = jnp.concatenate(
        [batch['goals'], batch['intermediate_goals'], batch['observations']], axis=0)

    g_aug = g + jax.random.normal(agent.rng, g.shape) * 0.01

    z1 = agent.network(g, method='get_task', params=network_params)
    z2 = agent.network(g_aug, method='get_task', params=network_params)

    z1 = z1 / jnp.linalg.norm(z1, axis=-1, keepdims=True)
    z2 = z2 / jnp.linalg.norm(z2, axis=-1, keepdims=True)

    logits = jnp.matmul(z1, z2.T) / temperature

    batch_size = g.shape[0]
    labels = jnp.arange(batch_size)

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, labels).mean()

    return loss, {"loss": loss}


def loss_fn(network_params, agent, batch):
    def phase_0(operand):
        network_params, agent, batch = operand
        info = all_info_dict.copy()

        loss, task_info = compute_continuous_contrastive_loss(
            agent, batch, network_params)

        for k, v in task_info.items():
            info[f'task_encoder/{k}'] = v

        return loss, info

    def phase_1(operand):
        network_params, agent, batch = operand
        info = all_info_dict.copy()  # Start with all keys and zero values

        task_embeddings = agent.network(
            batch['goals'], method='get_task', params=network_params)
        task = jax.lax.stop_gradient(task_embeddings)

        task_ens = jnp.expand_dims(task, axis=0)

        value_loss, value_info = compute_value_loss(
            agent, batch, network_params, task)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        all_obs = jnp.concatenate([
            batch['observations'],
            batch['intermediate_goals'],
        ], axis=0)

        # Raw phi embeddings, no cosine similarity.
        all_phis = agent.network(
            # [Ens, 2*B, Dim]
            all_obs, method='get_all_phis', params=network_params)

        phis, interm_phis = jnp.split(
            all_phis, 2, axis=1)  # Each is [Ens, B, Dim]

        hitting_times = batch['intermediate_distances']

        # Discounted hitting times
        if agent.config['discount'] == 1.0:
            hitting_times_discounted = hitting_times.astype(float)
        else:
            hitting_times_discounted = (
                1-agent.config['discount'] ** hitting_times.astype(float))/(1-agent.config['discount'] + 1e-8)

        displacement_s = interm_phis - phis
        step_progress_s = jnp.sum(displacement_s * task_ens, axis=-1)
        error_s = hitting_times_discounted - step_progress_s

        expectile_s = agent.config['HT_expectile']
        weight_s = jnp.where(error_s > 0, expectile_s, 1 - expectile_s)
        directed_step_loss_s = jnp.mean(weight_s * (error_s**2)) * 0.05

        directed_step_loss = directed_step_loss_s
        info['directed_step_loss_state'] = directed_step_loss_s

        loss = directed_step_loss + value_loss
        return loss, info

    def phase_2(operand):
        network_params, agent, batch = operand
        info = all_info_dict.copy()  # Start with all keys and zero values
        batch_size = batch['observations'].shape[0]
        rng = agent.rng
        rng, skill_rng = jax.random.split(rng)

        all_obs = jnp.concatenate([
            batch['observations'],
            batch['next_observations'],
        ], axis=0)

        all_phis = agent.network(all_obs, method='phi', params=network_params)

        phis, next_phis = jnp.split(all_phis, 2, axis=0)

        phis_no_grad = jax.lax.stop_gradient(phis)
        next_phis_no_grad = jax.lax.stop_gradient(next_phis)

        # Skill policy update
        batch['phis'] = phis_no_grad
        batch['next_phis'] = next_phis_no_grad
        random_skills = jax.random.normal(
            skill_rng, (batch_size, agent.config['skill_dim']))
        batch['skills'] = random_skills / \
            jnp.linalg.norm(random_skills, axis=-1, keepdims=True)
        batch['rewards'] = ((batch['next_phis'] - batch['phis'])
                            * batch['skills']).sum(axis=1)

        skill_value_loss, skill_value_info = compute_skill_value_loss(
            agent, batch, network_params)
        for k, v in skill_value_info.items():
            info[f'skill_value/{k}'] = v

        skill_critic_loss, skill_critic_info = compute_skill_critic_loss(
            agent, batch, network_params)
        for k, v in skill_critic_info.items():
            info[f'skill_critic/{k}'] = v

        skill_actor_loss, skill_actor_info = compute_skill_actor_loss(
            agent, batch, network_params)
        for k, v in skill_actor_info.items():
            info[f'skill_actor/{k}'] = v

        loss = skill_value_loss + skill_critic_loss + skill_actor_loss
        return loss, info

    operand = (network_params, agent, batch)

    phase_0_steps = agent.config['phase_0_steps']
    phase_1_steps = agent.config['phase_1_steps']

    # If step < phase 0 steps, run phase 0.
    # Else if step < (phase 0 + phase 1), run phase 1.
    # Else, run phase 2.
    loss, info = jax.lax.cond(
        agent.step < phase_0_steps,
        phase_0,
        lambda op: jax.lax.cond(
            agent.step < (phase_0_steps + phase_1_steps),
            phase_1,
            phase_2,
            op
        ),
        operand
    )

    return loss, info


class IELAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    network: TrainState
    config: dict = flax.struct.field(pytree_node=False)
    step: int = 0

    def update(agent, batch):
        new_target_params = jax.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (
                1 - agent.config['target_update_rate']), agent.network.params['networks_value'], agent.network.params['networks_target_value']
        )
        new_skill_target_params = jax.tree_map(
            lambda p, tp: p * agent.config['target_update_rate'] + tp * (
                1 - agent.config['target_update_rate']), agent.network.params['networks_skill_critic'], agent.network.params['networks_skill_target_critic']
        )

        new_network, info = agent.network.apply_loss_fn(
            loss_fn=partial(loss_fn, agent=agent, batch=batch), has_aux=True)

        params = unfreeze(new_network.params)
        params['networks_target_value'] = new_target_params
        params['networks_skill_target_critic'] = new_skill_target_params
        new_network = new_network.replace(params=freeze(params))
        new_rng, _ = jax.random.split(agent.rng)

        return agent.replace(network=new_network, rng=new_rng, step=agent.step + 1), info
    update = jax.jit(update)

    def get_loss_info(agent, batch):
        loss, info = loss_fn(agent.network.params, agent, batch)

        return info
    get_loss_info = jax.jit(get_loss_info)

    def sample_skill_actions(agent,
                             observations: np.ndarray,
                             skills: np.ndarray = None,
                             *,
                             seed: PRNGKey = None,
                             temperature: float = 1.0) -> jnp.ndarray:
        dist = agent.network(observations, skills,
                             temperature=temperature, method='skill_actor')
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions
    sample_skill_actions = jax.jit(sample_skill_actions)

    @jax.jit
    def get_phi(agent, s: np.ndarray) -> jnp.ndarray:
        phi = agent.network(s, method='phi')
        return phi

    @jax.jit
    def get_task(agent, s: np.ndarray) -> jnp.ndarray:
        task = agent.network(s, method='get_task')
        return task

    @jax.jit
    def get_all_phis(agent, s: np.ndarray) -> jnp.ndarray:
        phi = agent.network(s, method='get_all_phis')
        return phi


def create_learner(
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        lr: float = 3e-4,
        value_hidden_dims: Sequence[int] = (512, 512, 512),
        actor_hidden_dims: Sequence[int] = (512, 512, 512),
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.95,
        beta: float = 0.3,
        HT_expectile: float = 0.25,
        use_layer_norm: int = 1,
        skill_dim: int = 32,
        skill_expectile: float = 0.9,
        skill_temperature: float = 10,
        skill_discount: float = 0.99,
        encoder: str = None,
        phase_0_steps: int = 20000,
        phase_1_steps: int = 480000,
        **kwargs):

    print('Extra kwargs:', kwargs)

    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

    if encoder is not None:
        encoder_module = encoders[encoder]
    else:
        encoder_module = None

    value_def = GoalConditionedTopologicalPhiValue(
        hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, ensemble=True, skill_dim=skill_dim,
        beta=beta, encoder=encoder_module)

    task_def = TaskModel(
        hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, ensemble=False, skill_dim=skill_dim, encoder=encoder_module)

    skill_value_def = GoalConditionedValue(
        hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, ensemble=False, encoder=encoder_module)

    skill_critic_def = GoalConditionedCritic(
        hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, ensemble=True, encoder=encoder_module)

    skill_actor_def = Policy(actor_hidden_dims, action_dim=actions.shape[-1], log_std_min=-
                             5.0, state_dependent_std=False, tanh_squash_distribution=False, encoder=encoder_module)

    network_def = IELNetwork(
        networks={
            'value': value_def,
            'target_value': copy.deepcopy(value_def),
            'task': task_def,

            'skill_value': skill_value_def,
            'skill_target_value': copy.deepcopy(skill_value_def),
            'skill_critic': skill_critic_def,
            'skill_target_critic': copy.deepcopy(skill_critic_def),
            'skill_actor': skill_actor_def,
        },
    )
    network_tx = optax.adam(learning_rate=lr)
    network_params = network_def.init(
        value_key, observations, observations, actions, np.zeros((1, skill_dim)))['params']
    network = TrainState.create(network_def, network_params, tx=network_tx)
    params = unfreeze(network.params)
    params['networks_target_value'] = params['networks_value']
    params['networks_skill_target_critic'] = params['networks_skill_critic']
    network = network.replace(params=freeze(params))

    config = flax.core.FrozenDict(dict(
        discount=discount, target_update_rate=tau, expectile=expectile, beta=beta, HT_expectile=HT_expectile,
        skill_dim=skill_dim, skill_expectile=skill_expectile, skill_temperature=skill_temperature, skill_discount=skill_discount,
        phase_0_steps=phase_0_steps,
        phase_1_steps=phase_1_steps
    ))

    return IELAgent(rng, network=network, config=config)
