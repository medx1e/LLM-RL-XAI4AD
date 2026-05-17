import jax
import jax.numpy as jnp
from waymax import dynamics
from waymax.agents.expert import infer_expert_action
from waymax.datatypes import Action

from vmax.simulator import visualization


def run_and_log_scenario(env, scenario, step_fn) -> int:
    def _is_done(_env, simulator_state):
        termination = _env.termination(simulator_state)
        truncation = _env.truncation(simulator_state)

        return jnp.logical_or(termination, truncation)

    state = env.reset(scenario)

    imgs = []
    imgs.append(visualization.plot_input_agent(state, env))

    count = 0

    while not _is_done(env, state):
        count += 1
        state = step_fn(state)
        imgs.append(visualization.plot_input_agent(state, env))

    return imgs


def expert_step(env, state):
    actions = infer_expert_action(state, dynamics.InvertibleBicycleModel(normalize_actions=True))
    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
    action_sdc = jnp.take_along_axis(actions.data, sdc_idx[..., None], axis=0)

    action = jnp.squeeze(action_sdc, axis=0)
    action = Action(data=action, valid=jnp.ones_like(action[..., 0:1], dtype=jnp.bool_))
    action.validate()

    return env.step(state, action)
