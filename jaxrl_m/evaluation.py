from src.d4rl_utils import kitchen_render
from typing import Dict
import jax
import jax.numpy as jnp
import gym
import numpy as np
from collections import defaultdict
import time
from tqdm import trange
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, minimum_spanning_tree
from functools import partial


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wrapper that supplies a jax random key to a function (using keyword `seed`).
    Useful for stochastic policies that require randomness.

    Similar to functools.partial(f, seed=seed), but makes sure to use a different
    key for each new call (to avoid stale rng keys).

    """

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def env_reset(env_name, env, goal_info, base_observation, policy_type):
    observation, done = env.reset(), False
    if policy_type == 'random_skill' and 'antmaze' in env_name:
        observation[:2] = [20, 8]
        env.set_state(observation[:15], observation[15:])

    if 'antmaze' in env_name:
        goal = env.wrapped_env.target_goal
        obs_goal = np.concatenate([goal, base_observation[-27:]])
    elif 'kitchen' in env_name:
        if 'visual' in env_name:
            observation = kitchen_render(env)
            obs_goal = goal_info['ob']
        else:
            observation, obs_goal = observation[:30], observation[30:]
            obs_goal[:9] = base_observation[:9]
    else:
        raise NotImplementedError

    return observation, obs_goal


def env_step(env_name, env, action):
    if 'antmaze' in env_name:
        next_observation, reward, done, info = env.step(action)
    elif 'kitchen' in env_name:
        next_observation, reward, done, info = env.step(action)
        if 'visual' in env_name:
            next_observation = kitchen_render(env)
        else:
            next_observation = next_observation[:30]
    else:
        raise NotImplementedError

    return next_observation, reward, done, info


def get_frame(env_name, env):
    if 'antmaze' in env_name:
        size = 200
        cur_frame = env.render(mode='rgb_array', width=size,
                               height=size).transpose(2, 0, 1).copy()
    elif 'kitchen' in env_name:
        cur_frame = kitchen_render(env, wh=100).transpose(2, 0, 1)
    else:
        raise NotImplementedError
    return cur_frame


def add_episode_info(env_name, env, info, trajectory):
    if 'antmaze' in env_name:
        info['final_dist'] = jnp.linalg.norm(
            trajectory['next_observation'][-1][:2] - env.wrapped_env.target_goal)
    elif 'kitchen' in env_name:
        info['success'] = float(info['episode']['return'] == 4.0)
    else:
        raise NotImplementedError


@partial(jax.jit, static_argnums=(1, 2))
def greedy_dpp(phi_matrix, k, sigma=1.0):
    n, d = phi_matrix.shape
    diags = jnp.ones(n)
    c = jnp.zeros((k, n))

    phi_sq_norms = jnp.sum(phi_matrix**2, axis=1)

    def scan_body(carry, step_idx):
        diags, c = carry
        best_idx = jnp.argmax(diags)

        best_phi = phi_matrix[best_idx]
        best_phi_sq_norm = jnp.sum(best_phi**2)

        dist_sq = best_phi_sq_norm + phi_sq_norms - \
            2 * jnp.dot(phi_matrix, best_phi)
        L_row = jnp.exp(-dist_sq / (2 * sigma**2))

        v = L_row - jnp.dot(c.T, c[:, best_idx])
        new_c_row = v / jnp.sqrt(jnp.maximum(diags[best_idx], 1e-9))
        c = c.at[step_idx].set(new_c_row)

        new_diags = diags - new_c_row**2
        new_diags = new_diags.at[best_idx].set(-1e10)

        return (new_diags, c), best_idx

    (final_diags, final_c), selected_indices = jax.lax.scan(
        scan_body, (diags, c), jnp.arange(k))
    return selected_indices


def symmetric_graph(agent, planning_info):
    all_phis = jax.jit(lambda obs: jnp.array(agent.get_phi(obs)))(
        planning_info['examples']['observations'])
    coreset_idx = greedy_dpp(
        all_phis, k=planning_info['coreset_size'], sigma=planning_info['sigma'])

    planning_info['examples']['phis'] = all_phis[coreset_idx]
    ex_phis = planning_info['examples']['phis']
    epsilon = planning_info['epsilon']
    k_neighbors = planning_info['k_neighbors']
    INF = 1e9

    sq_norms = jnp.sum(ex_phis**2, axis=1)
    dist_matrix = jnp.sqrt(jnp.maximum(
        sq_norms[:, None] + sq_norms[None, :] -
        2 * jnp.dot(ex_phis, ex_phis.T),
        0.0
    ))

    print(
        f"Distance matrix stats: mean={jnp.mean(dist_matrix):.4f}, min={jnp.min(dist_matrix):.4f}, max={jnp.max(dist_matrix):.4f}")

    # MST Backbone
    dist_matrix_cpu = np.array(dist_matrix)
    mst_matrix = minimum_spanning_tree(dist_matrix_cpu)
    mst_adj = mst_matrix.toarray()
    mst_adj = np.maximum(mst_adj, mst_adj.T)  # Symmetrize MST

    # Epsilon mask (INF for non-edges)
    adj = jnp.where(dist_matrix < epsilon, dist_matrix, INF)

    # K-Nearest Neighbors
    neg_dist = -dist_matrix
    neg_dist = neg_dist.at[jnp.diag_indices(neg_dist.shape[0])].set(-1e10)
    _, topk_indices = jax.lax.top_k(neg_dist, k_neighbors)

    rows = jnp.arange(ex_phis.shape[0])[:, None]
    adj = adj.at[rows, topk_indices].min(dist_matrix[rows, topk_indices])

    # MST Edges ensure global connectivity
    u_idx, v_idx = np.where(mst_adj > 0)
    adj = adj.at[u_idx, v_idx].min(jnp.array(mst_adj[u_idx, v_idx]))

    adj = jnp.minimum(adj, adj.T)
    adj = adj.at[jnp.diag_indices(adj.shape[0])].set(0.0)

    sparse_ready_adj = jnp.where(adj == INF, 0.0, adj)
    graph = csr_matrix(np.array(sparse_ready_adj))
    planning_info['graph'] = graph


def get_all_pairs_costs(agent, coreset_obs, task, chunk_size=256):
    num_nodes = coreset_obs.shape[0]
    cost_matrix = jnp.zeros((num_nodes, num_nodes))

    def get_chunk_vals(s_chunk, all_obs):
        def row_fn(s):
            def single_cost(g):
                val = agent.network(
                    s[None], g[None], task[None], method='value')
                return -val[0, 0]

            return jax.vmap(single_cost)(all_obs)
        return jax.vmap(row_fn)(s_chunk)

    for i in range(0, num_nodes, chunk_size):
        s_chunk = coreset_obs[i: i + chunk_size]
        chunk_vals = get_chunk_vals(s_chunk, coreset_obs)
        cost_matrix = cost_matrix.at[i: i + chunk_size].set(chunk_vals)

    return cost_matrix


def asymmetric_graph(agent, planning_info, goal):
    ex_phis = planning_info['examples']['phis']
    epsilon = planning_info['epsilon']
    k = planning_info['k_neighbors']
    INF = 1e9

    task = agent.get_task(goal)

    cost_matrix = get_all_pairs_costs(
        agent, planning_info['examples']['coreset_observations'], task)

    print(
        f"mean cost: {jnp.mean(cost_matrix):.4f}, min cost: {jnp.min(cost_matrix):.4f}, max cost: {jnp.max(cost_matrix):.4f}")

    # Symmetrized Matrix for MST Skeleton
    # Uses average cost of (i->j and j->i) to find the best bidirectional bridges
    sym_cost_for_mst = (cost_matrix + cost_matrix.T) / 2.0

    mst_matrix = minimum_spanning_tree(np.array(sym_cost_for_mst))
    mst_indices = np.where(mst_matrix.toarray() > 0)

    # Epsilon masking
    adj = jnp.where(cost_matrix < epsilon, cost_matrix, INF)

    # K-NN masking (per row)
    neg_costs = -cost_matrix
    neg_costs = neg_costs.at[jnp.diag_indices(neg_costs.shape[0])].set(-1e10)
    _, topk_indices = jax.lax.top_k(neg_costs, k)

    rows = jnp.arange(ex_phis.shape[0])[:, None]
    adj = adj.at[rows, topk_indices].set(cost_matrix[rows, topk_indices])

    # Add MST edges to the graph in BOTH directions
    # Uses their original asymmetric costs
    u_idx, v_idx = mst_indices
    adj = adj.at[u_idx, v_idx].set(cost_matrix[u_idx, v_idx])
    adj = adj.at[v_idx, u_idx].set(cost_matrix[v_idx, u_idx])

    # Remove self-loops
    adj = adj.at[jnp.diag_indices(adj.shape[0])].set(0.0)
    sparse_ready_adj = jnp.where(adj == INF, 0.0, adj)

    graph = csr_matrix(np.array(sparse_ready_adj))
    planning_info['directed_graph_reversed'] = graph.transpose()


def asymmetric_distances(from_phi, to_phi, task, beta):
    diff = to_phi - from_phi
    dist = jnp.sqrt(jnp.maximum(
        jnp.sum(jnp.square(diff), axis=-1, keepdims=True), 1e-6))
    cos_sim = jnp.sum((diff / dist) * task, axis=-1, keepdims=True)
    v = dist * jnp.exp(beta * (1.0 - cos_sim))
    return jnp.squeeze(v, axis=-1)


def evaluate_with_trajectories(
        agent, env: gym.Env, goal_info, env_name, num_episodes, base_observation=None, num_video_episodes=0,
        policy_type='goal_skill', planning_info=None,
) -> Dict[str, float]:
    policy_fn = supply_rng(agent.sample_skill_actions)
    print(f"Evaluating with policy type: {policy_type}")

    if policy_type == 'goal_skill_planning':
        planning_info['examples']['phis'] = np.array(
            agent.get_phi(planning_info['examples']['observations']))

    if policy_type == 'graph_planning_symmetric':
        symmetric_graph(agent, planning_info)

    if policy_type == 'graph_planning_asymmetric':
        all_phis = jax.jit(lambda obs: jnp.array(agent.get_phi(obs)))(
            planning_info['examples']['observations'])
        coreset_idx = greedy_dpp(
            all_phis, k=planning_info['coreset_size'], sigma=planning_info['sigma'])
        print(
            f"Selected {len(np.unique(coreset_idx))} should be {planning_info['coreset_size']} indices: {coreset_idx}")
        planning_info['examples']['phis'] = all_phis[coreset_idx]
        planning_info['examples']['coreset_observations'] = planning_info['examples']['observations'][coreset_idx]

    trajectories = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_episodes + num_video_episodes):
        no_path_count = 0

        trajectory = defaultdict(list)

        observation, obs_goal = env_reset(
            env_name, env, goal_info, base_observation, policy_type)
        done = False

        render = []
        step = 0
        skill = None

        if policy_type == 'graph_planning_symmetric':
            ex_phis = planning_info['examples']['phis']
            phi_goal = agent.get_phi(np.array([obs_goal]))[0]
            goal_node_idx = np.argmin(
                np.linalg.norm(ex_phis - phi_goal, axis=-1))
            _, preds = shortest_path(
                csgraph=planning_info['graph'],
                directed=False,
                indices=goal_node_idx,  # SSSP from goal to everywhere
                return_predecessors=True
            )

        if policy_type == 'graph_planning_asymmetric':
            ex_phis = planning_info['examples']['phis']
            phi_goal = agent.get_phi(np.array([obs_goal]))[0]
            task = agent.get_task(obs_goal)
            goal_node_idx = np.argmin(
                asymmetric_distances(ex_phis, phi_goal, task, beta=planning_info['beta']))
            asymmetric_graph(agent, planning_info, obs_goal)
            _, preds = shortest_path(
                csgraph=planning_info['directed_graph_reversed'],
                directed=True,
                indices=goal_node_idx,  # SSSP from goal to everywhere
                return_predecessors=True
            )

        while not done:
            policy_obs = observation
            policy_goal = obs_goal

            if policy_type == 'goal_skill':
                phi_obs, phi_goal = agent.get_phi(
                    np.array([policy_obs, policy_goal]))
                skill = (phi_goal - phi_obs) / \
                    jnp.linalg.norm(phi_goal - phi_obs)
                action = policy_fn(observations=policy_obs,
                                   skills=skill, temperature=0.)
            elif policy_type == 'goal_skill_planning':
                phi_obs, phi_goal = agent.get_phi(
                    np.array([policy_obs, policy_goal]))

                for k in range(planning_info['num_recursions']):
                    ex_phis = planning_info['examples']['phis']
                    dists_s = jnp.linalg.norm(ex_phis - phi_obs, axis=-1)
                    dists_g = jnp.linalg.norm(ex_phis - phi_goal, axis=-1)
                    dists_diff = jnp.maximum(dists_s, dists_g)
                    way_idxs = jnp.argsort(dists_diff)
                    phi_goal = ex_phis[way_idxs[:planning_info['num_knns']]].mean(
                        axis=0)
                way_skill = (phi_goal - phi_obs) / \
                    jnp.linalg.norm(phi_goal - phi_obs)
                action = policy_fn(observations=policy_obs,
                                   skills=way_skill, temperature=0.)
            elif policy_type == 'graph_planning_symmetric':
                ex_phis = planning_info['examples']['phis']
                phi_obs = agent.get_phi(np.array([policy_obs]))[0]
                curr_node_idx = np.argmin(
                    np.linalg.norm(ex_phis - phi_obs, axis=-1))

                next_waypoint_idx = preds[curr_node_idx]
                if next_waypoint_idx != -9999 and curr_node_idx != goal_node_idx:
                    target_phi = ex_phis[next_waypoint_idx]
                else:
                    # If no path exists, head for goal
                    target_phi = phi_goal
                    no_path_count += 1

                way_skill = (target_phi - phi_obs) / \
                    (jnp.linalg.norm(target_phi - phi_obs) + 1e-8)
                action = policy_fn(observations=policy_obs,
                                   skills=way_skill, temperature=0.)

            elif policy_type == 'graph_planning_asymmetric':
                ex_phis = planning_info['examples']['phis']
                phi_obs = agent.get_phi(np.array([policy_obs]))[0]
                curr_node_idx = np.argmin(
                    asymmetric_distances(phi_obs, ex_phis, task, beta=planning_info['beta']))

                next_waypoint_idx = preds[curr_node_idx]
                if next_waypoint_idx != -9999 and curr_node_idx != goal_node_idx:
                    target_phi = ex_phis[next_waypoint_idx]
                else:
                    # If no path exists, head for goal
                    target_phi = phi_goal
                    no_path_count += 1

                way_skill = (target_phi - phi_obs) / \
                    (jnp.linalg.norm(target_phi - phi_obs) + 1e-8)
                action = policy_fn(observations=policy_obs,
                                   skills=way_skill, temperature=0.)

            else:
                raise NotImplementedError

            action = np.array(action)
            next_observation, reward, done, info = env_step(
                env_name, env, action)
            step += 1

            # Render
            if i >= num_episodes and step % 3 == 0:
                cur_frame = get_frame(env_name, env)
                render.append(cur_frame)
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                skill=skill,
                info=info,
            )
            if i < num_episodes:
                add_to(trajectory, transition)
                add_to(stats, flatten(info))
            observation = next_observation
        if no_path_count > 0:
            print(f"Episode {i}: No path found for {no_path_count} steps.")
        if i < num_episodes:
            add_episode_info(env_name, env, info, trajectory)
            add_to(stats, flatten(info, parent_key="final"))
            trajectories.append(trajectory)
        else:
            renders.append(np.array(render))

    scalar_stats = {}
    for k, v in stats.items():
        scalar_stats[k] = np.mean(v)
    return scalar_stats, trajectories, renders


class EpisodeMonitor(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.get_normalized_score(
                        info["episode"]["return"]) * 100.0
                )

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()
