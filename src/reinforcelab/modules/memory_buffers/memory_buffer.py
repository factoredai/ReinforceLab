from abc import ABC, abstractmethod
from reinforcelab.modules.experience import Experience, BatchExperience


class MemoryBuffer(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        """Creates a Memory Buffer instance
        Args:
            config (dict): includes information required for the instance to work
        """

    @abstractmethod
    def add(self, experience: Experience):
        """Adds an experience tuple to the memory buffer

        Args:
            experience (Experience): An experience instance
        """

    @abstractmethod
    def sample(self) -> BatchExperience:
        """Retrieves a set of experience tuples from the memory buffer

        Args:
            batch_size (int): The number of samples to retrieve in a single batch
        """

    @abstractmethod
    def all(self) -> BatchExperience:
        """Retrieves all experiences from the memory buffer

        Returns:
            BatchExperience: All experiences as a BatchExperience
        """

    @abstractmethod
    def __len__(self):
        """Number of samples in the memory buffer
        """
