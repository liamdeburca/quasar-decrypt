def _round(n:int) -> callable:
    from numpy import round
    
    func = lambda val: '{}'.format(round(val, n))

    return func

def _scientific(n:int) -> callable:
    
    match n:
        case 1:
            func = lambda val: '{:.1e}'.format(val)
        case 2:
            func = lambda val: '{:.2e}'.format(val)
        case 3:
            func = lambda val: '{:.3e}'.format(val)
        case 4:
            func = lambda val: '{:.4e}'.format(val)
        case 5: 
            func = lambda val: '{:.5e}'.format(val)
        case 6:
            func = lambda val: '{:.6e}'.format(val)

    return func

def format_float_to_str(val, 
                        n_commas,
                        scientific:bool=False):
    
    if scientific:
        func = _scientific(n_commas)
    else:
        func = _round(n_commas)

    if isinstance(val, (float, int)):
        out = func(val)
    elif isinstance(val, list):
        out = list(map(func, val))

    return out
    
###

def _pad_string(length:int, filler:str, where:str) -> callable:
    match where:
        case 'left':
            func = lambda string: string.ljust(length, filler)
        case 'right':
            func = lambda string: string.rjust(length, filler)

    return func
    
def format_string_to_length(string, 
                            length:int,
                            filler:str=' ', 
                            where:str='left'):
    
    func = _pad_string(length, filler, where)

    if isinstance(string, str):
        out = func(string)
    elif isinstance(string, list):
        out = list(map(func, string))
    else:
        out = format_string_to_length(
            str(string), 
            length, 
            filler=filler, 
            where=where,
        )

    return out

###

def get_column_minwidth(column:list, mode:str='tab'):
    from numpy import max

    max_length = max(list(map(len, column)))

    match mode:
        case 'tab':
            w = 4 * (max_length // 4 + 1)
        case _:
            w = max_length

    return w

from numpy import ndarray

def format_column(x:ndarray, header:str, n_commas:int=2):
    x_str = format_float_to_str(list(x), n_commas)
    
    column = [header] + x_str
    w = get_column_minwidth(column)
    column = format_string_to_length(column, w)

    return column