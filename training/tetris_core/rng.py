"""Unsigned 32-bit Mulberry32 matching game.js, including JS Math.imul behavior."""

MASK = 0xffffffff


class Mulberry32:
    def __init__(self, seed):
        if type(seed) is not int or not 0 <= seed <= MASK:
            raise ValueError("Seed must be an unsigned 32-bit integer")
        self.state = seed

    def random(self):
        self.state = (self.state + 0x6d2b79f5) & MASK
        value = self.state
        value = ((value ^ (value >> 15)) * (value | 1)) & MASK
        value ^= (value + (((value ^ (value >> 7)) * (value | 61)) & MASK)) & MASK
        return ((value ^ (value >> 14)) & MASK) / 4294967296
