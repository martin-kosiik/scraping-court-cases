#import numpy as np

def flatten_lists(the_list):
    #import numpy as np
    if not isinstance(the_list, list):
        out_list = the_list
    elif len(the_list) == 0:
        out_list = np.nan
    elif len(the_list) == 1:
        out_list = the_list[0]
    else:
        out_list = the_list
    return out_list

def unique_list(the_list):
    if not isinstance(the_list, list):
        out_list = the_list
    else:
        out_list = list(dict.fromkeys(the_list))
    return out_list
