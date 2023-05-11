def f_priority(g, h):
    return g + h


def fw_priority(w):
    def fw_grounded(g, h):
        return g + w * h


def h_priority(g, h):
    return h
