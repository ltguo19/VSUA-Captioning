import os
import torch
import cPickle
import logging

def save_nets_structure(nets, opt):
    path = os.path.join(opt.checkpoint_path, 'models.txt')
    if type(nets) is list:
        nets = { 'Net_'+str(i): net for i, net in enumerate(nets)}
    with open(path, 'w') as f:
        for name, net in nets.items():
            if net is not None:
                f.write("Name: {}\n{}\n".format(name, net))


# save
def save_checkpoint(models, optimizers, infos, is_best, opt):
    "save model, optimizer, and infos"
    assert type(models) is dict and type(optimizers) is dict, \
        "models and optimizers should be dict"
    def save(prefix):
        model_params = {name: model.state_dict() for name, model in models.items()}
        torch.save(model_params, prefix+'model.pth')
        optimizer_params = {name: optimizer.state_dict() for name, optimizer in optimizers.items()}
        torch.save(optimizer_params, prefix+'optimizer.pth')
        with open(prefix+'infos.pkl', 'wb') as f:
            cPickle.dump(infos, f)

    logger = logging.getLogger('__main__')
    # save current and override
    prefix = os.path.join(opt.checkpoint_path, '')
    logger.info("Saving checkpoint to {}".format(prefix + 'model.pth'))
    save(prefix)
    # save best
    if is_best:
        prefix = os.path.join(opt.checkpoint_path, 'best_')
        logger.info("Saving best checkpoint")
        save(prefix)

# load
def load_checkpoint(models, optimizers, opt):
    logger = logging.getLogger('__main__')
    # check compatibility if training is continued from previously saved model
    if opt.resume_from is None or os.path.isdir(opt.resume_from):
        logger.info("resume_from not set, training from scratch")
        return False
    prefix = 'best_' if opt.resume_from_best else ''
    flag = True

    # load model
    path = os.path.join(opt.resume_from, prefix + 'model.pth')
    if os.path.isfile(path):
        logger.info("Loading models from %s" % path)
        params = torch.load(path)
        for name, model in models.items():
            model.load_state_dict(params[name])
    else:
        logger.warning("Fail to load model")
        flag = False

    # load optimizer
    path = os.path.join(opt.resume_from, prefix + 'optimizer.pth')
    if os.path.isfile(path):
        logger.info("Loading optimizers from %s" % path)
        params = torch.load(path)
        for name, optim in optimizers.items():
            optim.load_state_dict(params[name])
    else:
        logger.warning("Fail to load optimizer")
        flag = False

    return flag


def load_info(opt):
    logger = logging.getLogger('__main__')
    # check compatibility if training is continued from previously saved model
    if opt.resume_from is None or os.path.isdir(opt.resume_from):
        logger.info("resume_from not set, not loadding infos")
        return {}
    prefix = 'best_' if opt.resume_from_best else ''
    path = os.path.join(opt.resume_from, prefix + 'infos.pkl')
    if os.path.isfile(path):
        logger.info("Loading infos from %s" % path)
        # open old infos and check if models are compatible
        with open(os.path.join(path)) as f:
            infos = cPickle.load(f)
        saved_model_opt = infos['opt']
        need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
        for checkme in need_be_same:
            assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
    else:
        logger.warning("Fail to load infos")
        infos = {}

    return infos
