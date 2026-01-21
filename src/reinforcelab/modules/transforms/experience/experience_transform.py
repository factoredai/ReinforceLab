from abc import ABC, abstractmethod

from reinforcelab.modules.experience import Experience


class ExperienceTransform(ABC):
    @abstractmethod
    def __call__(self, experience: Experience) -> Experience:
        """Applies a transformation to an experience instance

        Args:
            experience (Experience): Experience instance to be transformed

        Returns:
            (Experience): A transformed version of the passed Experience
        """
