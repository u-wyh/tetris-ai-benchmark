"""The fixed 1840-action Benchmark v1.0 placement address space."""

ACTION_COUNT = 1840


def encode_action(hold, rotation, x, y):
    if (not all(type(value) is int for value in (hold, rotation, x, y))
            or hold not in (0, 1) or rotation not in range(4)
            or x not in range(10) or y not in range(-3, 20)):
        raise ValueError("Action coordinates are outside the 1840-action space")
    return (((hold * 4 + rotation) * 10 + x) * 23 + y + 3)


def decode_action(action):
    if type(action) is not int or not 0 <= action < ACTION_COUNT:
        raise ValueError("Action ID must be an integer from 0 to 1839")
    y = action % 23 - 3
    rest = action // 23
    x = rest % 10
    rest //= 10
    return {"hold": rest // 4, "rotation": rest % 4, "x": x, "y": y}
