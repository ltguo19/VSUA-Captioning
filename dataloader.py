from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
import logging
import pickle
import torch
import torch.utils.data as data

class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split=='train', self.opt.loader_num_workers, self.opt)
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.logger = logging.getLogger('__main__')
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.geometry_relation = opt.geometry_relation

        # feature related options
        self.use_att = getattr(opt, 'use_att', True)
        self.use_fc = getattr(opt, 'use_fc', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)

        # data dir
        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir

        # scene graph data
        self.sg_data_dir = opt.sg_data_dir
        self.sg_geometry_dir = self.opt.sg_geometry_dir
        self.sg_vocab = np.load(opt.sg_vocab_path)[()]

        # load the json file which contains additional information about the dataset
        self.logger.info('DataLoader loading json file: %s'% opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        self.logger.info('vocab size is %d' %self.vocab_size)
        
        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]
        self.logger.info('max sequence length in data is %d' % self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0]
        self.logger.info('read %d image features' % (self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)
        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        for split in self.split_ix.keys():
            self.logger.info('assigned %d images to split %s' % (len(self.split_ix[split]), split))

        # load the width and height of images
        if self.use_box:
            self.logger.info('Loading sg_box_info')
            self.sg_box_info = pickle.load(open(opt.sg_box_info_path))

        self._prefetch_process = {} # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train', self.opt.loader_num_workers, self.opt)
            # Terminate the child process when the parent exists
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        fc_batch = [] # np.ndarray((batch_size * seq_per_img, self.opt.fc_feat_size), dtype = 'float32')
        att_batch = [] # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')
        sg_batch = []
        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'float32')
        infos = []
        gts = []
        wrapped = False

        for i in range(batch_size):
            # fetch image
            tmp_fc, tmp_att, tmp_sg, \
                ix, tmp_wrapped = self._prefetch_process[split].get()
            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)
            sg_batch.append(tmp_sg)

            label_batch[i * seq_per_img : (i + 1) * seq_per_img, 1 : self.seq_length + 1] = self.get_captions(ix, seq_per_img)

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
        
            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        data = {}
        data['fc_feats'] = np.stack(reduce(lambda x,y:x+y, [[_]*1 for _ in fc_batch]))
        # merge att_feats
        max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros([len(att_batch), max_att_len, att_batch[0].shape[1]], dtype = 'float32')
        for i in range(len(att_batch)):
            data['att_feats'][i, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        # if data['att_masks'].sum() == data['att_masks'].size:
        #     data['att_masks'] = None

        data['labels'] = label_batch # np.vstack(label_batch)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, data['labels'])))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch

        data['gts'] = gts # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        data['sg_data'] = self.batch_sg(sg_batch, max_att_len)

        return data


    def batch_sg(self, sg_batch, max_att_len):
        "batching object, attribute, and relationship data"
        obj_batch = [_['obj'] for _ in sg_batch]
        attr_batch = [_['attr'] for _ in sg_batch]
        rela_batch = [_['rela'] for _ in sg_batch]
        sg_data = {}

        # obj labels, shape: (B, No, 1)
        sg_data['obj_labels'] = np.zeros([len(obj_batch), max_att_len, self.opt.num_obj_label_use], dtype = 'int')
        for i in range(len(obj_batch)):
            sg_data['obj_labels'][i, :obj_batch[i].shape[0]] = obj_batch[i]

        # attr labels, shape: (B, No, 3)
        sg_data['attr_labels'] = np.zeros([len(attr_batch), max_att_len, self.opt.num_attr_label_use], dtype = 'int')
        for i in range(len(attr_batch)):
            sg_data['attr_labels'][i, :attr_batch[i].shape[0]] = attr_batch[i]
        # obj and attr share the same mask as att_feats

        # rela
        max_rela_len = max([_['edges'].shape[0] for _ in rela_batch])
        sg_data['rela_edges'] = np.zeros([len(rela_batch), max_rela_len, 2], dtype = 'int')
        if self.geometry_relation:
            sg_data['rela_feats'] = np.zeros([len(rela_batch), max_rela_len, self.opt.geometry_rela_feat_dim], dtype = 'float32')
        else:
            sg_data['rela_feats'] = np.zeros([len(rela_batch), max_rela_len], dtype='int')
        # rela_masks, because no all items in rela_edges and rela_feats are meaningful
        sg_data['rela_masks'] = np.zeros(sg_data['rela_edges'].shape[:2], dtype='float32')

        for i in range(len(rela_batch)):
            sg_data['rela_edges'][i, :rela_batch[i]['edges'].shape[0]] = rela_batch[i]['edges']
            sg_data['rela_feats'][i, :rela_batch[i]['edges'].shape[0]] = rela_batch[i]['feats']
            sg_data['rela_masks'][i, :rela_batch[i]['edges'].shape[0]] = 1

        return sg_data


    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index #self.split_ix[index]
        image_id = str(self.info['images'][ix]['id'])
        if self.use_att:
            att_feat = np.load(os.path.join(self.input_att_dir, image_id + '.npz'))['feat']
            # Reshape to K x C
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            if self.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
            if self.use_box:
                box_feat = self.get_box_feat(image_id)
                att_feat = np.hstack([att_feat, box_feat])
                # sort the features by the size of boxes
                # att_feat = np.stack(sorted(att_feat, key=lambda x:x[-1], reverse=True))
        else:
            att_feat = np.zeros((1,1,1))
        fc_feat = np.load(os.path.join(self.input_fc_dir, image_id + '.npy'))

        sg_data = self.get_graph_data(index)

        return (fc_feat,
                att_feat,
                sg_data,
                ix)


    def get_graph_data(self, index):
        image_id = str(self.info['images'][index]['id'])
        sg_use = np.load(self.sg_data_dir + image_id + '.npy')[()]
        if self.geometry_relation:
            geometry_path = os.path.join(self.sg_geometry_dir, image_id + '.npy')
            rela = np.load(geometry_path)[()] # dict contains keys of edges and feats
        else:
            # if the relation of an image is empty, then fill in it with <0, 0, 'near'> to avoid problems
            if sg_use['rela_matrix'].shape[0] == 0:
                sg_use['rela_matrix'] = np.array([0, 0, self.sg_vocab['w2i']['near']], dtype=sg_use['rela_matrix'].dtype)

            # shape (Nr, 3), column index 0,1 is edge, index 2 is relation label
            triplet = sg_use['rela_matrix']
            rela = {}
            rela['edges'] = triplet[:, 0:2]
            rela['feats'] = triplet[:, 2]

        obj = sg_use['obj_attr'][:, 1:1+self.opt.num_obj_label_use]  # shape (No, ?)
        attr = sg_use['obj_attr'][:, 4:4+self.opt.num_attr_label_use] # shape (No, ?)

        sg_data = {'obj': obj, 'attr': attr, 'rela': rela}
        return sg_data


    def get_box_feat(self, image_id):
        image = self.sg_box_info[int(image_id)]
        x1, y1, x2, y2 = np.hsplit(image['boxes'], 4)
        h, w = image[int(image_id)]['image_h'], image[int(image_id)]['image_w']
        iw, ih = x2 - x1 + 1, y2 - y1 + 1
        box_feat = np.hstack((0.5 * (x1 + x2) / w, 0.5 * (y1 + y2) / h, iw / w, ih / h, iw * ih / (w * h)))
        if self.norm_box_feat:
            box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
        return box_feat


    def __len__(self):
        return len(self.info['images'])


class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False, num_workers = 4, opt=None):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.opt =opt
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle
        self.num_workers = num_workers

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            sampler=SubsetSampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=self.num_workers, # 4 is usually enough
                                            worker_init_fn=None,
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[-1] == ix, "ix not equal"

        return tmp + [wrapped]