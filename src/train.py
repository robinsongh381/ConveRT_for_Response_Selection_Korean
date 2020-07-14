from __future__ import absolute_import, division, print_function, unicode_literals
import argparse, random, os, logging, glob
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange, tqdm_notebook, tnrange

import constant as config
from trainer import ConveRTTrainer
from model import ConveRTDualEncoder
from criterion import ConveRTLoss
from dataset import ConveRTDataset, convert_pad_collate, MappingDataset, mapping_pad_collate
from logger import logger, init_logger

device = config.device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # log file
    parser.add_argument('-log_folder', default='/epoch_{}_batch_{}_layer_{}_inner_head_{}_outer_head+{}_loss_type_{}_kobert_{}')
    parser.add_argument('-log_dir', default='./experiment')
    # `parser.add_argument("-num_tr_sent", default=10, type=int, choices=[5,10])
    parser.add_argument('-loss_type', default='dot')
    parser.add_argument('-add_response', default='True', choices=['True','False'])
    parser.add_argument('-kobert', default=True)
    # model config
    parser.add_argument("-enc_layer", default=config.encoder_layer, type=int)
    parser.add_argument("-epoch", default=config.epoch, type=int)
    parser.add_argument("-batch_size", default=config.batch_size, type=int)

    
    args = parser.parse_args()
    print(args)
    
    log_path = args.log_dir + args.log_folder.format(args.epoch, args.batch_size, args.enc_layer, config.nhead_inner, config. nhead_outter, args.loss_type, args.kobert)
    print('log_path : {}'.format(log_path))
    init_logger(log_path,'/log/log.txt')
    
    # checkpoint_manager = CheckpointManager(args.log_path)
    tb_writer = SummaryWriter('{}/runs'.format(log_path))
    
    # Load dataset
    print('Data loading...')
    train_dataset = ConveRTDataset('tr')
    train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=convert_pad_collate,
                              drop_last=True,
                              num_workers=0)
    
    valid_dataset = ConveRTDataset('test')
    valid_dataloader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=convert_pad_collate,
                              drop_last = True,
                              num_workers=0)
    
    
    # Load intent label - indicies pair
    mapping_dataset = ConveRTDataset('tr')
    mapping_dataloader = DataLoader(dataset=mapping_dataset,
                              batch_size=1, # args.batch_size,
                              shuffle=False,
                              collate_fn=convert_pad_collate,
                              drop_last=True,
                              num_workers=0)
#     mapping_dataset = MappingDataset()
#     mapping_dataloader = DataLoader(dataset=mapping_dataset,
#                               batch_size=1, # args.batch_size,
#                               shuffle=False,
#                               collate_fn=mapping_pad_collate,
#                               drop_last=True,
#                               num_workers=0)
    # Model $ Criterion
    print('Model loading...')
    model = ConveRTDualEncoder(config, args, logger)
    logger.info(model)
    logger.info(config)
    # model.to(device)
    
    criterion = ConveRTLoss(config=config, loss_type=args.loss_type)
    # criterion.to(devcie)
    
    trainer = ConveRTTrainer(
                    args=args,
                    config=config,
                    model=model,
                    criterion=criterion,
                    train_dataloader=train_dataloader,
                    valid_dataloader=valid_dataloader,
                    mapping_dataloader=mapping_dataloader,
                    logger=logger,
                    save_path=log_path,
                    tb_writer=tb_writer)
    
    print('Train Start...')
    trainer.train()
    
    print('Train finished !')
    
    print('Prining f1 score...')
    # trainer.compute_f1_score()