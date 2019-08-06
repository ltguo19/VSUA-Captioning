import argparse
import os
from utils.helper import str2bool


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='test',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--gpus', type=str, default='0', help='set CUDA_VISIBLE_DEVICES')
    # Data input settings
    parser.add_argument('--input_json', type=str, default='data/cocotalk.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_fc_dir', type=str, default='data/cocobu_fc',
                    help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='data/cocobu_att',
                    help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_label_h5', type=str, default='data/cocotalk_label.h5',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--train_only', type=int, default=0,
                    help='if true then use 80k, else use 110k')
    parser.add_argument('--cached_tokens', type=str, default='coco-train-idxs',
                    help='Cached token file for calculating cider score during self critical training.')
    parser.add_argument('--loader_num_workers', type=int, default=4,
                    help='num of processes to use for BlobFetcher')


    # load model and settings
    parser.add_argument('--resume_from', type=str, default=None,
                    help="continuing training from this experiment id")
    parser.add_argument('--resume_from_best', type=str2bool, default=False,
                    help='resume from best model, True: use best_model.pth, False: use model.pth')
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')

    # Model settings
    parser.add_argument('--caption_model', type=str, default="vsua",
                    help='model type: [vsua]')
    parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
    parser.add_argument('--logit_layers', type=int, default=1,
                    help='number of layers in the RNN')


    # feature manipulation
    parser.add_argument('--norm_att_feat', type=int, default=0,
                    help='If normalize attention features')
    parser.add_argument('--use_box', type=str2bool, default=False,
                    help='If use box features')
    parser.add_argument('--norm_box_feat', type=int, default=0,
                    help='If use box, do we normalize box feature')
    parser.add_argument('--use_bn', type=int, default=0,
                    help='If 1, then do batch_normalization first in att_embed, if 2 then do bn both in the beginning and the end of att_embed')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=-1,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=10,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--self_critical_after', type=int, default=-1,
                    help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')

    # learning rate
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1,
                    help='at what epoch to start decaying learning rate? (-1 = dont)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8,
                    help='every how many iterations thereafter to drop LR?(in epoch)')

    # scheduled sampling
    parser.add_argument('--scheduled_sampling_start', type=int, default=-1,
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5,
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                    help='Maximum scheduled sampling prob.')


    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=5000,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=-1,
                    help='how often to save a model checkpoint (in iterations)? (-1 = every epoch)')
    parser.add_argument('--checkpoint_root', type=str, default='log',
                    help='root directory to store checkpointed models')
    parser.add_argument('--checkpoint_path', type=str, default='',
                    help='directory to store current checkpoint, \
                         if not set, it will be assigned as (args.checkpoint_root, args.id) by default. ')
    parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L/SPICE? requires coco-caption code from Github.')
    parser.add_argument('--log_loss_every', type=int, default=10,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

    # Reward
    parser.add_argument('--cider_reward_weight', type=float, default=1,
                    help='The reward weight from cider')
    parser.add_argument('--bleu_reward_weight', type=float, default=0,
                    help='The reward weight from bleu4')


    # VSUA related inputs
    parser.add_argument('--sg_vocab_path', type=str, default='data/coco_pred_sg_rela.npy',
                        help='path to the vocab file, containing vocabularies of object, attribute, relationships')
    parser.add_argument('--sg_data_dir', type=str, default='data/coco_img_sg/',
                        help='path to the scene graph data directory, containing numpy files about the '
                             'labels of object, attribute, and semantic relationships for each image')
    parser.add_argument('--sg_geometry_dir', type=str, default='data/geometry-iou0.2-dist0.5-undirected/',
                        help='directory of geometry edges and features')
    parser.add_argument('--sg_box_info_path', type=str, default='data/vsua_box_info.pkl',
                        help='path to the pickle file containing the width and height infos of images')
    parser.add_argument('--num_obj_label_use', type=int, default=1,
                        help='number of object labels to use')
    parser.add_argument('--num_attr_label_use', type=int, default=3,
                        help='number of attribute labels to use')
    parser.add_argument('--geometry_rela_feat_dim', type=int, default=8,
                        help='dim of geometry relationship features')


    # VSUA model settings
    parser.add_argument('--vsua_use', type=str, default='oar',
                    help='which types of visual semantic units to contain, o: obj, a: attr, r: rela')
    parser.add_argument('--geometry_relation', type=str2bool, default=True,
                        help='type of relationship to use, True: geometry relationship, False: semantic relationship')
    parser.add_argument('--rela_gnn_type', type=int, default=0,
                        help='rela gcn type')
    parser.add_argument('--sg_label_embed_size', type=int, default=128,
                    help='graph embedding_size of obj, attr, rela')

    args = parser.parse_args()

    if args.checkpoint_path=='':
        args.checkpoint_path = os.path.join(args.checkpoint_root, args.id)
    if not os.path.exists(args.checkpoint_root):
        os.mkdir(args.checkpoint_root)
    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)

    if args.resume_from:
        path = os.path.join(args.checkpoint_root, args.resume_from)
        assert os.path.exists(path), "%s not exists" %args.resume_from

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print("[INFO] set CUDA_VISIBLE_DEVICES = %s" % args.gpus)

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    return args