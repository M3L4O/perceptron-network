def binary_step(u: float) -> int:
    if u < 0:
        return 0
    else:
        return 1


def bipolar_binary_step(u: float) -> int:
    if u < 0:
        return -1
    else:
        return 1
