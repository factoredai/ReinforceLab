from .experience import Experience

class PriorityExperience(Experience):
    def __init__(self, state, action, reward, next_state, done, error):
        super().__init__(state, action, reward, next_state, done)
        self.error = error

    def __iter__(self):
        super().__iter__()
        yield self.error