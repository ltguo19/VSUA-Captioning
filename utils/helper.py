from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception("value not allowed")


def str2list(v):
    return v.split(",")

