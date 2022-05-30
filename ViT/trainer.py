import torch
import torch.nn as nn
import torch.optim as optim
import warmup_scheduler
import numpy as np
import random

import os
from utils import progress_bar
from da import CutMix, MixUp



class Trainer(object):
    def __init__(self, model, args):
        self.device = args.device[0]
        self.model = model
        self.cutmix = args.cutmix
        self.mixup = args.mixup
        self.get_li = args.get_li
        print(model)
        if args.cutmix:
            self.cutmix_ = CutMix(args.size, beta=1.)
        if args.mixup:
            self.mixup_ = MixUp(alpha=1.)
        
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.max_epochs, eta_min=args.min_lr)
        
        if args.warmup_epoch:
            self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=args.warmup_epoch, after_scheduler=self.base_scheduler)
        else:
            self.scheduler = self.base_scheduler
        #self.scaler = torch.cuda.amp.GradScaler()

        self.epochs = args.max_epochs
        self.criterion = nn.CrossEntropyLoss()

        self.num_steps = 0
        self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.
        
        self.total = self.correct = self.test_total = self.test_correct = 0.
    def _train_one_step(self, batch):
        
        self.model.train()
        img, label = batch
        self.num_steps += 1
        img, label = img.to(self.device), label.to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.cutmix or self.mixup:
            if self.cutmix:
                img, label, rand_label, lambda_= self.cutmix_((img, label))
            elif self.mixup:
                if np.random.rand() <= 0.8:
                    img, label, rand_label, lambda_ = self.mixup_((img, label))
                else:
                    img, label, rand_label, lambda_ = img, label, torch.zeros_like(label), 1.
            pre = torch.cuda.memory_allocated(self.device)/1024/1024
            out, li = self.model(img)
            post = torch.cuda.memory_allocated(self.device)/1024/1024
            loss = self.criterion(out, label)*lambda_ + self.criterion(out, rand_label)*(1.-lambda_)
        else:
            pre = torch.cuda.memory_allocated(self.device)/1024/1024
            out, li = self.model(img)
            post = torch.cuda.memory_allocated(self.device)/1024/1024
            loss = self.criterion(out, label)
            
        act_mem = post-pre
        loss.backward()
        self.optimizer.step()
        
        loss += loss.item()
        _, predicted = out.max(1)
        self.total += img.size(0)
        self.correct += predicted.eq(label).sum().item()

        return loss, 100.*self.correct/self.total, self.correct, self.total, li, act_mem  

    # @torch.no_grad
    def _test_one_step(self, batch):
        
        self.model.eval()
        img, label = batch
        img, label = img.to(self.device), label.to(self.device)

        with torch.no_grad():
            out, li = self.model(img)
            loss = self.criterion(out, label)

        loss += loss.item()
        _, predicted = out.max(1)
        self.test_total += img.size(0)
        self.test_correct += predicted.eq(label).sum().item()
        
        return loss, 100.*self.test_correct/self.test_total, self.test_correct, self.test_total

    def train(self, train_dl, test_dl, experiment_name):
        best_acc=0
        max_mem=0
        step=0
        result = []
        total_li = []
        for epoch in range(1, self.epochs+1):
            print('\nEpoch: %d' % epoch)
            for batch_idx, batch in enumerate(train_dl):
                loss, acc, correct, total, li, act_mem = self._train_one_step(batch)
                progress_bar(batch_idx, len(train_dl), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (loss, acc, correct, total))
                if step % len(train_dl) == 0:
                    total_li += random.sample(li, 1500) 
                step += 1
                
            self.scheduler.step()
            
            for batch_idx, batch in enumerate(test_dl):
                test_loss, test_acc, test_correct, test_total = self._test_one_step(batch)
                
                progress_bar(batch_idx, len(test_dl), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss, test_acc, test_correct, test_total))
            
            if test_acc > best_acc:
                print('Saving..')
                state = {
                    'net': self.model.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')    
                torch.save(state, './checkpoint/' + experiment_name + '.pth')
                best_acc = test_acc
                
            if act_mem > max_mem:
                max_mem = act_mem
            print('best accuracy : ' + str(round(best_acc, 2)), 'training_memory : ' + str(round(max_mem, 2)) + ' Mib')
            
            if not os.path.isdir('results'):
                os.mkdir('results')
            result.append([acc, test_acc, loss.item(), test_loss.item()])
            self.test_correct = self.test_total = 0.
            self.correct = self.total = 0.
            np.save("results/" + experiment_name, result)
            if self.get_li:
                np.save("results/" + experiment_name + '_li', total_li)
            
        return 

    def eval(self, test_dl):
        for batch_idx, batch in enumerate(test_dl):
            test_loss, test_acc, test_correct, test_total = self._test_one_step(batch)
            
            progress_bar(batch_idx, len(test_dl), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss, test_acc, test_correct, test_total))
        return 
