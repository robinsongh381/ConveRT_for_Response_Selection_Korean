from __future__ import absolute_import, division, print_function, unicode_literals

import glob, os
import torch
from torch.optim import Adam
from pytorch_transformers import AdamW, WarmupLinearSchedule
from tqdm import tqdm, trange, tqdm_notebook, tnrange
from sklearn.metrics import f1_score, precision_score
import torch_optimizer as optim

class ConveRTTrainer:
    def __init__(self,
                 args,
                 config,
                 model, 
                 criterion, 
                 train_dataloader, 
                 valid_dataloader,
                 mapping_dataloader,
                 logger,
                 save_path,
                 tb_writer):
        
        self.args = args
        self.config = config
        self.model = model
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.mapping_dataloader = mapping_dataloader
        self.logger = logger
        self.save_path = save_path
        self.tb_writer = tb_writer
        
        self.t_total = len(self.train_dataloader)*self.args.epoch
        self.device = self.config.device
        # self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.optimizer = optim.RAdam(self.model.parameters(), lr= self.config.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        # self.scheduler = WarmupLinearSchedule(self.optimizer, 0.1*self.t_total, self.t_total)
           
        self.global_step = 0
        self.best_eval_acc = 0.6
        
        self.tr_correct_cnt = 0.0
        self.tr_correct_cnt_recall = 0.0
        self.tr_total_cnt = 0.0
        self.tr_loss = 0.0
        self.eval_total_loss = 0.0
        self.best_eval_acc = 0.0
        

    def train(self, do_eval=True, do_save=True):
        for epoch in range(self.args.epoch):
            self.train_epoch(epoch)
            self.evaluation(epoch)
            # self.save_model(epoch) 
        self.response_to_vector()
        self.compute_f1_score()
                 

    def train_epoch(self, epoch):       
        self.model.to(self.device)
        self.model.train() 
        self.optimizer.zero_grad()

        train_loader = self.train_dataloader
        for step, batch in enumerate(train_loader):
            self.global_step += 1 #step
            
            self.model.zero_grad()   
            context, response = map(lambda x: torch.from_numpy(x).to(self.device), batch[:2])
            true_label = list(batch[2])
            context_embed, response_embed = self.model(context, response)
            loss, correct_cnt, total_cnt, pred_label = self.criterion(context_embed, response_embed)
            
            self.tr_correct_cnt += correct_cnt[0]
            self.tr_correct_cnt_recall += correct_cnt[1]
            self.tr_total_cnt += total_cnt
            self.tr_loss += loss.item()  
            
            loss.backward()
            self.optimizer.step()
            self.model.zero_grad()
            # self.scheduler.step()   
            
        tr_avg_acc = float(self.tr_correct_cnt)/self.tr_total_cnt
        tr_avg_recall = float(self.tr_correct_cnt_recall)/self.tr_total_cnt
        tr_avg_fscore = 2*tr_avg_acc*tr_avg_recall / (tr_avg_recall+tr_avg_acc)
        tr_avg_loss = self.tr_loss/self.tr_total_cnt                

        self.logger.info('epoch : {}, step : {} /{}, tr_avg_loss: {:.4f}, tr_avg_fscore: {:.2%}'.format(
                          epoch+1, self.global_step, self.t_total, tr_avg_loss, tr_avg_fscore)) 
    
        self.tr_avg_acc =  self.tr_correct_cnt / self.tr_total_cnt
        self.tr_avg_loss =  self.tr_loss / self.tr_total_cnt
                       
        
    def evaluation(self, epoch):  
        self.model.eval()
        eval_correct_cnt, eval_correct_cnt_recall, eval_total_cnt = 0,0,0
        eval_loss = 0.0
        
        for step, batch in enumerate(self.valid_dataloader):
            sent, response = map(lambda x: torch.from_numpy(x).to(self.device), batch[:2])
            true_label = list(batch[2])
            context_embed, reply_embed = self.model(sent, response)
            loss, correct_cnt, total_cnt, pred_label = self.criterion(context_embed, reply_embed)
                
            eval_correct_cnt += correct_cnt[0]
            eval_correct_cnt_recall += correct_cnt[1] ##
            eval_total_cnt += total_cnt
            eval_loss += loss.item()

        self.eval_avg_acc = float(eval_correct_cnt) / eval_total_cnt
        self.eval_avg_acc_recall = float(eval_correct_cnt_recall) / eval_total_cnt ##
        self.eval_avg_fscore = 2*self.eval_avg_acc*self.eval_avg_acc_recall / (self.eval_avg_acc+self.eval_avg_acc_recall)
        
        self.eval_loss = eval_loss / eval_total_cnt
        self.eval_total_loss += self.eval_loss
        self.eval_avg_loss = self.eval_total_loss / (epoch+1)
        
        self.logger.info('epoch : {}, step : {} /{}, eval_loss: {:.3f}, eval_avg_loss: {:.4f}, eval_avg_fscore: {:.2%}'.format(
            epoch+1, self.global_step, self.t_total, self.eval_loss, self.eval_avg_loss, self.eval_avg_fscore))
        
        if self.eval_avg_fscore > self.best_eval_acc:
            self.best_eval_acc = self.eval_avg_fscore
            self.best_epoch = epoch +1  
 

    def response_to_vector(self):
        self.model.eval()
        self.response_matrix = torch.tensor([]).to(self.device)
        last_label = None
        
        for batch in self.mapping_dataloader:
            label = batch[2]
            response_to_idx = torch.from_numpy(batch[1]).to(self.device)
            
            if label != last_label:
                last_label = label

                with torch.no_grad():
                    reponse_embed = self.model(response_input = response_to_idx, response_only=True)
                    self.response_matrix = torch.cat([self.response_matrix, reponse_embed])

        # torch.save(self.response_matrix, './experiment/response_matirx.pt')
        # self.logger.info('Intent mapping tensor saved !')
        
        
    def compute_f1_score(self):
        self.model.eval()
        
        # For train data
        gold_tr=[]
        pred_tr=[]
        for batch in self.train_dataloader:
            gold_tr += list(batch[2])
            context = torch.from_numpy(batch[0]).to(self.device)
            context_embed = self.model(context, context_only=True)
            
            # Compute (batch x num_intent) similarity table
            sim_matrix = torch.matmul(context_embed, self.response_matrix.transpose(0,1))
            pred_label = sim_matrix.argmax(-1)
            pred_tr += pred_label.tolist()
        train_f1 = f1_score(gold_tr, pred_tr, average='weighted')
        self.logger.info('train_f1 score : {:.2%}'.format(train_f1))    
        
        # For test data
        gold_eval=[]
        pred_eval=[]        
        for batch in self.valid_dataloader:
            gold_eval += list(batch[2])
            context = torch.from_numpy(batch[0]).to(self.device)
            context_embed = self.model(context, context_only=True)
            
            # Compute (batch x num_intent) similarity table
            sim_matrix = torch.matmul(context_embed, self.response_matrix.transpose(0,1))
            pred_label = sim_matrix.argmax(-1)
            pred_eval += pred_label.tolist()
            

        eval_f1 = f1_score(gold_eval, pred_eval, average='weighted')
        self.logger.info('eval_f1 score : {:.2%}'.format(eval_f1))
        self.logger.info(' ')
        # eval_precision = precision_score(gold_eval, pred_eval, average='weighted')
        # self.logger.info('eval_precision score : {:.2%}'.format(eval_precision))
        
        
    def save_model(self, epoch):
        
        if self.eval_avg_acc > self.best_eval_acc:
            self.best_eval_acc = self.eval_avg_acc
        
            self.model.to(torch.device('cpu'))
            state = {'epoch': epoch+1,
                     'model_state_dict': self.model.state_dict(),
                     'opt_state_dict': self.optimizer.state_dict()}

            save_model_path = '{}/epoch_{}_step_{}_tr_acc_{:.3f}_tr_loss_{:.3f}_eval_acc_{:.3f}_eval_loss_{:.3f}.pt'.format(
                        self.save_path, epoch+1, self.global_step, self.tr_avg_acc,self.tr_avg_loss, self.eval_avg_acc,  self.eval_avg_loss)
                
                
            # Delte previous checkpoint
            if len(glob.glob(self.save_path+'/epoch*.pt'))>0:
                os.remove(glob.glob(self.save_path+'/epoch*.pt')[0])
            torch.save(state, save_model_path)
            self.logger.info(' Model saved to {}'.format(save_model_path))
            
            
    def write_to_tb(self):
        self.tb_writer.add_scalars('loss', {'train': self.tr_avg_loss, 'val': self.eval_avg_loss}, self.global_step)
        self.tb_writer.add_scalars('acc', {'train': self.tr_avg_acc, 'val': self.eval_avg_acc}, self.global_step) 