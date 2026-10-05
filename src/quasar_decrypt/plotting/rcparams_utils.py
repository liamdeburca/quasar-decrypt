from matplotlib import rcParams


def get_fontsize() -> float:
    return rcParams["font.size"]

def get_linewidth() -> float:
    return rcParams["lines.linewidth"]

# X-Ticks

def get_xtick_major_size() -> float:
    return rcParams["xtick.major.size"]

def get_xtick_minor_size() -> float:
    return rcParams["xtick.minor.size"]

def get_xtick_major_width() -> float:
    return rcParams["xtick.major.width"]

def get_xtick_minor_width() -> float:
    return rcParams["xtick.minor.width"]

# Y-Ticks

def get_ytick_major_size() -> float:
    return rcParams["ytick.major.size"]

def get_ytick_minor_size() -> float:
    return rcParams["ytick.minor.size"]

def get_ytick_major_width() -> float:
    return rcParams["ytick.major.width"]

def get_ytick_minor_width() -> float:
    return rcParams["ytick.minor.width"]

###

def get_xmargin() -> float:
    return rcParams["axes.xmargin"]

def get_ymargin() -> float:
    return rcParams["axes.ymargin"]

def get_zmargin() -> float:
    return rcParams["axes.zmargin"]