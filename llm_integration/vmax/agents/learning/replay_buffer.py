# Copyright 2025 Valeo.


"""Replay buffer module."""

import flax
import jax
import jax.numpy as jnp
from jax import flatten_util

from vmax.agents import datatypes


@flax.struct.dataclass
class ReplayBufferState:
    """Contains data related to a replay buffer."""

    data: jax.Array
    insert_position: int
    sample_position: int
    key: jax.Array


class ReplayBuffer:
    """Replay buffer."""

    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        samples_size: int,
        dummy_data_sample: datatypes.RLPartialTransition,
    ):
        """Initialize the replay buffer.

        Args:
            buffer_size: The size of the replay buffer.
            batch_size: The size of the batch to sample from the replay buffer.
            samples_size: The size of the samples to store in the replay buffer.
            dummy_data_sample: A sample of the data to be stored in the replay buffer.

        """
        self._flatten_fn = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])
        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        self._unflatten_fn = jax.vmap(self._unflatten_fn)
        data_size = len(dummy_flatten)

        self._samples_size = samples_size
        self._data_shape = (buffer_size, data_size)
        self._data_dtype = jnp.float16
        self._batch_size = batch_size
        self._size = buffer_size

    def init(self, key: jax.Array) -> ReplayBufferState:
        """Initialize the replay buffer.

        Args:
            key: A random key.

        Returns:
            The initial state of the replay buffer.

        """
        return ReplayBufferState(
            data=jnp.zeros(self._data_shape, self._data_dtype),
            insert_position=0,
            sample_position=0,
            key=key,
        )

    def insert(self, buffer_state: ReplayBufferState, samples: datatypes.RLTransition) -> ReplayBufferState:
        """Insert transitions into the replay buffer.

        Args:
            buffer_state: The current buffer state.
            samples: A batch of RL transitions.

        Returns:
            The updated buffer state.

        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"buffer_state.data.shape ({buffer_state.data.shape}) doesn't \
                match the expected value ({self._data_shape})",
            )

        # Flatten the samples and cast to 16 bits for storage.
        new_samples = self._flatten_fn(samples).astype(self._data_dtype)
        samples_size = len(new_samples)

        # Current buffer state
        data = buffer_state.data
        insert_idx = buffer_state.insert_position
        sample_idx = buffer_state.sample_position

        # Update the buffer and the control numbers
        data = jax.lax.dynamic_update_slice_in_dim(data, new_samples, insert_idx, axis=0)
        insert_idx = (insert_idx + samples_size) % self._size
        sample_idx = jnp.minimum(buffer_state.sample_position + samples_size, self._size)

        return buffer_state.replace(data=data, insert_position=insert_idx, sample_position=sample_idx)

    def sample(self, buffer_state: ReplayBufferState) -> tuple[ReplayBufferState, datatypes.RLTransition]:
        """Sample a batch of transitions from the replay buffer.

        Args:
            buffer_state: The current buffer state.

        Returns:
            A tuple containing the updated buffer state and sampled transitions.

        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})",
            )

        key, sample_key = jax.random.split(buffer_state.key)

        candidates = jax.random.randint(sample_key, (self._batch_size,), 0, buffer_state.sample_position)
        mask = candidates != buffer_state.insert_position - 1
        idx = jnp.where(mask, candidates, buffer_state.insert_position - 2)

        # Cast data back to 32 bits after sampling.
        current_data = jnp.take(buffer_state.data, idx, axis=0, unique_indices=True).astype(jnp.float32)
        current_batch = self._unflatten_fn(current_data)

        # Get next observation from the next index in the buffer
        next_idx = (idx + self._samples_size) % buffer_state.sample_position
        next_data = jnp.take(buffer_state.data, next_idx, axis=0, unique_indices=True).astype(jnp.float32)
        next_batch = self._unflatten_fn(next_data)

        # Build full RL transitions with the next_observation
        transitions = datatypes.RLTransition(
            observation=current_batch.observation,
            action=current_batch.action,
            reward=current_batch.reward,
            flag=current_batch.flag,
            next_observation=next_batch.observation,
            done=current_batch.done,
            extras=current_batch.extras,
        )

        return buffer_state.replace(key=key), transitions
