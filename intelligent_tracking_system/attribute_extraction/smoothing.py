from collections import defaultdict, deque
from statistics import mode, mean
import numpy as np

class AttributeSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=window_size)))

    def update(self, person_id: str, attributes: dict) -> dict:
        smoothed = {}

        for key, value in attributes.items():
            # Convert numpy bool to Python bool
            if isinstance(value, (np.bool_, bool)):
                value = bool(value)
            self.history[person_id][key].append(value)

            values = list(self.history[person_id][key])
            if isinstance(value, (int, float)):
                smoothed[key] = round(mean(values), 2)
            elif isinstance(value, bool):
                smoothed[key] = values.count(True) > values.count(False)
            else:
                try:
                    smoothed[key] = mode(values)
                except:
                    smoothed[key] = values[-1]  # fallback to latest

        return smoothed
