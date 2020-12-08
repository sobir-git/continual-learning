import time

from utils import Timer


class TestTimer:
    times = [0.001, 0.01, 0.1, 0.5, 1]
    _eps = 0.002

    def almost_equal(self, a, b):
        return abs(a - b) < self._eps

    def test_timer_ctx_mangager(self):
        timer = Timer()
        for v in self.times:
            with timer:
                time.sleep(v)
            assert self.almost_equal(timer._values[-1], v)

    def test_timer(self):
        timer = Timer()
        for v in self.times:
            timer.start()
            time.sleep(v)
            timer.finish()
            assert self.almost_equal(timer.values[-1], v)

        assert len(timer.values) == len(self.times)

    def test_timer_attach(self):
        f = lambda v: time.sleep(v)
        timer = Timer()
        f = timer.get_timed_callable(f)
        for v in self.times:
            f(v)
            assert self.almost_equal(timer._values[-1], v)

    def test_timer_get_timed_generator(self):
        timer = Timer()
        gen = (time.sleep(v) for v in self.times)
        timed_gen = timer.get_timed_generator(gen)
        gen_values = list(timed_gen)
        assert not timer._running
        assert len(timer.values) == len(gen_values) == len(self.times)
        for i in range(len(self.times)):
            assert self.almost_equal(self.times[i], timer.values[i])

    def test_timer_get_timed_generator_multiple(self):
        gen1 = (time.sleep(v) for v in self.times)
        gen2 = (time.sleep(v) for v in self.times)

        timer = Timer()
        t_gen1 = timer.get_timed_generator(gen1)
        t_gen2 = timer.get_timed_generator(gen2)

        for i in range(len(self.times)):
            next(t_gen1), next(t_gen2)
            assert self.almost_equal(self.times[i], timer.values[i * 2])
            assert self.almost_equal(self.times[i], timer.values[i * 2 + 1])
