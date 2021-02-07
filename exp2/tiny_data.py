import numpy as np
import torchvision.datasets as td


class TinyCIFAR10(td.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        maxidx = self.data.shape[0]
        _rs = np.random.RandomState(seed=0)
        indices = _rs.randint(0, maxidx, size=800)
        self.targets = [self.targets[i] for i in indices]
        self.data = self.data[indices]
        assert len(set(self.targets)) == 10


td.TinyCIFAR10 = TinyCIFAR10
