import time
from utils import Timer


class TestTimer():
    values = [0.001, 0.01, 0.1, 0.5, 1]
    _eps = 0.001

    def almost_equal(self, a, b):
        return abs(a - b) < self._eps

    def test_timer_ctx_mangager(self):
        timer = Timer()
        for v in self.values:
            with timer:
                time.sleep(v)
            assert self.almost_equal(timer._values[-1], v)

    def test_timer(self):
        timer = Timer()
        for v in self.values:
            timer.start()
            time.sleep(v)
            timer.finish()
            assert self.almost_equal(timer.values[-1], v)

        assert len(timer.values) == len(self.values)

    def test_timer_attach(self):
        f = lambda v: time.sleep(v)
        timer = Timer()
        f = timer.attach(f)
        for v in self.values:
            f(v)
            assert self.almost_equal(timer._values[-1], v)
