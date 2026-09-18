"""Fail-closed ordering for the local two-runtime RGB bridge."""


class ReplayState:
    def __init__(self):
        self.version = 0
        self.phase = 'idle'

    def require(self, version, phase):
        if version != self.version or self.phase != phase:
            raise ValueError(f'Stale/out-of-order request: {version}/{phase}; expected {self.version}/{self.phase}')

    def encoded(self, version):
        self.require(version, 'idle')
        self.phase = 'encoded'

    def backward_ready(self, version):
        self.require(version, 'encoded')
        self.phase = 'backward_ready'

    def committed(self, version):
        self.require(version, 'backward_ready')
        self.version += 1
        self.phase = 'idle'
