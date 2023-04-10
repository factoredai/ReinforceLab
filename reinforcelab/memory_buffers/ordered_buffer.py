from collections import deque

from .memory_buffer import MemoryBuffer
from reinforcelab.experience import Experience, BatchExperience


class OrderedBuffer(MemoryBuffer):
    """An ordered buffer returns the latest batch of experience in the
    order it was added. a FIFO queue implementation of the experience replay.
    This could be useful for agents that use a trace to estimate the value of a state,
    like SARSA agents.
    """

    def __init__(self, config):
        self.experience_buffer = deque(maxlen=config['max_size'])
        self.batch_size = config['batch_size']

    def add(self, experience: Experience):
        """Adds an experience tupple to the memory buffer

        Args:
            experience (Experience): An experience instance
        """
        self.experience_buffer.append(experience)

    def sample(self) -> BatchExperience:
        """Retrieves the latest batch of experience in an ordered sequence

        Returns:
            BatchExperience: a batch of experiences
        """
        if len(self) < self.batch_size:
            raise RuntimeError("Not enough experience to sample a batch")
        experiences = list(self.experience_buffer)[-self.batch_size:]
        return BatchExperience(experiences)

    def __len__(self):
        return len(self.experience_buffer)
