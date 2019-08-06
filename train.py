from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import opts
from models import setup

import eval_utils as eval_utils
import misc.utils as utils
from utils.logger import *
from utils.load_save import *
from misc.rewards_graph import init_scorer, get_self_critical_reward
from dataloader import *


def train(opt):
    # Deal with feature things before anything
    opt.use_att = utils.if_use_att(opt.caption_model)
    opt.use_fc = utils.if_use_fc(opt.caption_model)

    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length


    infos = load_info(opt)
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = infos.get('val_result_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)

    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    # Define and load model, optimizer, critics
    decoder = setup(opt).train().cuda()
    crit = utils.LanguageModelCriterion().cuda()
    rl_crit = utils.RewardCriterion().cuda()
    optimizer = utils.build_optimizer(decoder.parameters(), opt)
    models = {'decoder': decoder}
    optimizers = {'decoder': optimizer}
    save_nets_structure(models, opt)
    load_checkpoint(models, optimizers, opt)


    epoch_done = True
    sc_flag = False
    while True:
        if epoch_done:
            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                decoder.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False
            epoch_done = False

        # 1. fetch a batch of data from train split
        data = loader.get_batch('train')
        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp
        sg_data = {key: data['sg_data'][key] if data['sg_data'][key] is None \
            else torch.from_numpy(data['sg_data'][key]).cuda() for key in data['sg_data']}

        # 2. Forward model and compute loss
        torch.cuda.synchronize()
        optimizer.zero_grad()
        if not sc_flag:
            out = decoder(sg_data, fc_feats, att_feats, labels, att_masks)
            loss = crit(out, labels[:, 1:], masks[:, 1:])
        else:
            gen_result, sample_logprobs, core_args = decoder(sg_data, fc_feats, att_feats, att_masks, opt={'sample_max': 0, 'return_core_args': True}, mode='sample')
            reward = get_self_critical_reward(decoder, core_args, sg_data, fc_feats, att_feats, att_masks, data, gen_result, opt)
            loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())

        # 3. Update model
        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.item()
        torch.cuda.synchronize()

        # Update the iteration and epoch
        iteration += 1
        # Write the training loss summary
        if (iteration % opt.log_loss_every == 0):
            # logging log
            logger.info("{} ({}), loss: {:.3f}".format(iteration, epoch, train_loss))
            tb.add_values('loss', {'train': train_loss}, iteration)

        if data['bounds']['wrapped']:
            epoch += 1
            epoch_done = True

        # Make evaluation and save checkpoint
        if (opt.save_checkpoint_every >0  and iteration % opt.save_checkpoint_every == 0) or (opt.save_checkpoint_every == -1 and epoch_done):
            # eval model
            eval_kwargs = {'split': 'val',
                           'dataset': opt.input_json,
                           'expand_features': False}
            eval_kwargs.update(vars(opt))
            predictions, lang_stats = eval_utils.eval_split(decoder, loader, eval_kwargs)
            # log val results
            if not lang_stats is None:
                logger.info("Scores: {}".format(lang_stats))
                tb.add_values('scores', lang_stats, epoch)
            val_result_history[epoch] = {'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            current_score = 0 if lang_stats is None else lang_stats['CIDEr']
            best_flag = False
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True

            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['vocab'] = loader.get_vocab()
            infos['val_result_history'] = val_result_history

            save_checkpoint(models, optimizers,
                            infos, best_flag, opt)

        # Stop if reaching max epochs
        if epoch > opt.max_epochs and opt.max_epochs != -1:
            break


opt = opts.parse_opt()
logger = define_logger(opt)
tb = MyTensorboard(opt)
train(opt)
