import torch
from typing import List
from .experience import Experience

class BatchExperience(Experience):
    def __init__(self, experiences: List[Experience]):
        self.experiences = experiences

        # Concatenate contents of all experiences
        self.state = torch.vstack([exp.state for exp in self.experiences])
        self.action = torch.vstack([exp.action for exp in self.experiences])
        self.reward = torch.vstack([exp.reward for exp in self.experiences]).float()
        self.next_state = torch.vstack([exp.next_state for exp in self.experiences]).float()
        self.done = torch.vstack([exp.done for exp in self.experiences]).float()

    def __iter__(self):
        yield self.state
        yield self.action
        yield self.reward
        yield self.next_state
        yield self.done
