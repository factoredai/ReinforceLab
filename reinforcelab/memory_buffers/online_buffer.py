from .memory_buffer import MemoryBuffer
from reinforcelab.experience import Experience, BatchExperience


class OnlineBuffer(MemoryBuffer):
    """Memory buffer for online learning. It doesn't actually provide storage, and instead
    just keeps a copy of the latest experience added.
    """

    def __init__(self):
        self.experience = None

    def add(self, experience: Experience):
        self.experience = experience

    def sample(self) -> BatchExperience:
        if self.experience is not None:
            return BatchExperience([self.experience])
        else:
            raise RuntimeError("No experience has been added to the buffer")

    def __len__(self):
        return int(self.experience is not None)
