import os
import dill
from torch import Tensor

from reinforcelab.agents import BaseAgent
from reinforcelab.brains import Brain
from reinforcelab.update_estimators import UpdateEstimator
from reinforcelab.action_selectors import ActionSelector
from reinforcelab.memory_buffers import MemoryBuffer

class ActorCriticAgent(BaseAgent):
    def __init__(
            self,
            local_actor: Brain,
            target_actor: Brain,
            local_critic: Brain,
            target_critic: Brain,
            update_estimator: UpdateEstimator,
            action_selector: ActionSelector,
            memory_buffer: MemoryBuffer, 
            update_every=1
        ):
        self.local_actor = local_actor
        self.target_actor = target_actor
        self.local_critic = local_critic
        self.target_critic = target_critic
        self.estimator = update_estimator
        self.action_selector = action_selector
        self.memory_buffer = memory_buffer
        self.update_every = update_every
        self.update_step = 0

    def act(self, state: Tensor, **kwargs) -> Tensor:
        pred_action = self.local_actor(state)
        return self.action_selector(pred_action, **kwargs)

    def update(self, experience):
        self.memory_buffer.add(experience)
        if self.update_step % self.update_every == 0:
            try:
                batch = self.memory_buffer.sample()
            except RuntimeError:
                # If the batch can't be obtained, skip the update proc
                return
            actor_pred, actor_target,  = self.estimator(batch)
            self.local_actor.update(batch, pred, target)
            self.target_actor.update_from(self.local_actor)

            self.local_critic.update(batch, pred, target)
            self.target_critic.update_from(self.local_critic)
        self.update_step += 1

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, "checkpoint.dill")
        with open(filepath, "wb") as f:
            dill.dump(self, f)

    def load(self, path):
        filepath = os.path.join(path, "checkpoint.dill")
        with open(filepath, "rb") as f:
            loaded_agent = dill.load(f)

        self.__dict__.clear()
        self.__dict__.update(loaded_agent.__dict__)