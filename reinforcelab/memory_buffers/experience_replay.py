import random
from collections import deque

from reinforcelab.memory_buffers.memory_buffer import MemoryBuffer
from reinforcelab.experience import Experience, BatchExperience

class ExperienceReplay(MemoryBuffer):
    def __init__(self, config):
        self.experience_buffer = deque(maxlen=config['max_size'])

    def add(self, experience: Experience):
        """Adds an experience tuple to the memory buffer

        Args:
            experience (Experience): An experience instance
        """
        self.experience_buffer.append(experience)

    def sample(self, batch_size):
        """Retrieves a random set of experience tuples from the memory buffer

        Args:
            batch_size (int): The number of samples to retrieve in a single batch

        Returns:
            (BatchExperience): A batch of experiences
        """
        experiences = random.sample(list(self.experience_buffer), batch_size)
        return BatchExperience(experiences)

    def __len__(self):
        return len(self.experience_buffer)