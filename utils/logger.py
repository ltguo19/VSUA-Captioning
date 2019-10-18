import os
import logging
try:
    import tensorboardX as tb
except ImportError:
    print("[Warning] tensorboardX is not installed")
    tb = None


def define_logger(opt, id=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    id = opt.id if id is None else id

    logfile = os.path.join(opt.checkpoint_path, opt.id + '.log')
    fh = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s ["+id+"] %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class MyTensorboard():
    def __init__(self, opt):
        self.writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    # add scalar
    def add_value(self, key, value, iteration):
        if self.writer:
            self.writer.add_scalar(key, value, iteration)

    # add dict
    def add_values(self, main_tag, tag_scalar_dict, iteration):
        if self.writer:
            for name, value in tag_scalar_dict.items():
                key = main_tag+'/'+name
                self.writer.add_scalar(key, value, iteration)
