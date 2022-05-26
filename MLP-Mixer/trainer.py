import torch
import torch.nn as nn
import torch.optim as optim
import warmup_scheduler
import numpy as np
from criterions import LabelSmoothingCrossEntropyLoss

import os
from utils import rand_bbox, progress_bar
import random

class Trainer(object):
    def __init__(self, model, args):
        self.device = args.device[0]
        self.clip_grad = args.clip_grad
        self.cutmix_beta = args.cutmix_beta
        self.cutmix_prob = args.cutmix_prob
        self.model = model
        self.get_li = args.get_li
        if args.optimizer=='sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer=='adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        else:
            raise ValueError(f"No such optimizer: {self.optimizer}")

        if args.scheduler=='step':
            self.base_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[args.epochs//2, 3*args.epochs//4], gamma=args.gamma)
        elif args.scheduler=='cosine':
            self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        else:
            raise ValueError(f"No such scheduler: {self.scheduler}")


        if args.warmup_epoch:
            self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=args.warmup_epoch, after_scheduler=self.base_scheduler)
        else:
            self.scheduler = self.base_scheduler
        #self.scaler = torch.cuda.amp.GradScaler()

        self.epochs = args.epochs
        self.criterion = LabelSmoothingCrossEntropyLoss(args.num_classes, smoothing=args.label_smoothing)

        self.num_steps = 0
        self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.
        
        self.total = self.correct = self.test_total = self.test_correct = 0.
    def _train_one_step(self, batch):
        
        self.model.train()
        img, label = batch
        self.num_steps += 1
        img, label = img.to(self.device), label.to(self.device)
        
        self.optimizer.zero_grad()
        r = np.random.rand(1)
        if self.cutmix_beta > 0 and r < self.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(self.cutmix_beta, self.cutmix_beta)
            rand_index = torch.randperm(img.size(0)).to(self.device)
            target_a = label
            target_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            # compute output
            pre_allocated = torch.cuda.memory_allocated(self.device) / 1024 /1024
            out, li = self.model(img)
            post_allocated = torch.cuda.memory_allocated(self.device) / 1024 /1024
            
            loss = self.criterion(out, target_a) * lam + self.criterion(out, target_b) * (1. - lam)
        else:
            # compute output
            pre_allocated = torch.cuda.memory_allocated(self.device) / 1024 /1024
            out, li = self.model(img)
            post_allocated = torch.cuda.memory_allocated(self.device) / 1024 /1024
            
            loss = self.criterion(out, label)
        
        act_mem = post_allocated-pre_allocated
        loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        #self.scaler.step(self.optimizer)
        #self.scaler.update()
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
        best_acc = 0
        max_mem = 0
        step = 0
        result = []
        total_li = []
        for epoch in range(1, self.epochs+1):
            print('\nEpoch: %d' % epoch)
            for batch_idx, batch in enumerate(train_dl):
                loss, acc, correct, total, li, act_mem = self._train_one_step(batch)
                progress_bar(batch_idx, len(train_dl), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (loss, acc, correct, total))
                if step % len(train_dl) == 0:
                    total_li += random.sample(li, 1000) 
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
            np.save("results2/" + experiment_name, result)
            if self.get_li:
                np.save("results2/" + experiment_name + '_li', total_li)
            
        return 

    def eval(self, test_dl):
        for batch_idx, batch in enumerate(test_dl):
                test_loss, test_acc, test_correct, test_total = self._test_one_step(batch)
                
                progress_bar(batch_idx, len(test_dl), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss, test_acc, test_correct, test_total))    
        return