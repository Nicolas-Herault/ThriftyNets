from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from utils.early_stopping import EarlyStopping
from utils.decorators import *
from utils.score.score import score2019

from dataloader import get_dataloaders
from mask import Mask


class Trainer():

    def __init__(self, model, dataloader_config, train_config):
        self.model = model # model must be an instance of the Model class
        self.trainloader, self.testloader = self._init_dataloaders(dataloader_config)
        self.config = train_config
        self.writer = self._init_writer()
        self.early_stopping = self._init_early_stopping()
        self.use_binary_connect = self._init_binary_connect()
        self.use_pruning = self._init_pruning()
        self.state = {'train_loss': 0,
                      'train_acc': 0,
                      'test_loss': 0,
                      'test_acc': 0,
                      'best_acc': 0,
                      'epoch': 0,
                      'score_param': 1,
                      'score_op': 1}


    def __str__(self):
        title = 'Training settings  :' + '\n' + '\n'
        dataset     = 'Dataset...................:  ' + self.model.summary['dataset'] + '\n'
        net         = 'Net.......................:  ' + self.model.summary['net'] + '\n'
        optimizer   = 'Optimizer.................:  ' + self.model.summary['optimizer'] + '\n'
        scheduler   = 'Learning Rate Scheduler...:  ' + self.model.summary['scheduler'] + '\n'
        nb_epochs   = 'Number of epochs..........:  ' + str(self.config['nb_epochs']) + '\n'
        use_bc      = 'Use Binary Connect........:  ' + str(self.use_binary_connect) + '\n'
        use_pruning = 'Use Soft Pruning..........:  ' + str(self.use_pruning) + '\n'
        model_summary = dataset + net + optimizer + scheduler
        train_summary = nb_epochs + use_bc + use_pruning
        return (80*'_' + '\n' + title + model_summary + train_summary + 80*'_')


    def _init_dataloaders(self, dataloader_config):
        return get_dataloaders(dataloader_config)


    def _init_writer(self):
        path = self.config['tensorboard_path'] + self.model.summary['net']
        writer = torch.utils.tensorboard.SummaryWriter(log_dir = path)
        return writer


    def _init_early_stopping(self):
        early_stopping = EarlyStopping(patience=self.config['patience'],
                                       delta = self.config['delta'],
                                       verbose=self.config['verbose'])
        early_stopping.set_checkpoints(self.model.summary,
                                       self.config['checkpoints_path'])
        return early_stopping


    def _init_binary_connect(self):
        if self.config['use_binary_connect']:
            self.binary_connect = self.model.binary_connect()
            return True
        else:
            return False


    def _init_pruning(self):
        pruning_config = self.config['pruning']
        if pruning_config['use_pruning']:
            self.mask = Mask(self.model.net, pruning_config, self.config['nb_epochs'])
            self.mask.net = self.model.net
            self.mask.init_mask(0) # epoch number is 0 here
            self.mask.do_mask()
            self.model.net = self.mask.net
            return True
        else:
            return False



    def train(self):
        self.model.net.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, labels in tqdm(self.trainloader):
            inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)
            if self.use_binary_connect:
                self.binary_connect.binarization()
            self.model.optimizer.zero_grad()
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = self.model.net(inputs)
            loss = self.model.criterion(outputs, labels)
            loss.backward()
            if self.use_binary_connect:
                self.binary_connect.clip()
            self.model.optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_acc = 100.*correct/total
        return train_loss, train_acc


    def test(self):
        self.model.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(self.testloader):
                inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)
                inputs, labels = Variable(inputs), Variable(labels)
                if self.use_binary_connect:
                    self.binary_connect.binarization()
                outputs = self.model.net(inputs)
                loss = self.model.criterion(outputs, labels)
                if self.use_binary_connect:
                    self.binary_connect.clip()
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_acc = 100.*correct/total
        return test_loss, test_acc


    def prune(self, epoch):
        self.mask.net = self.model.net
        self.mask.verbose()
        self.mask.init_mask(epoch)
        self.mask.do_mask()
        self.mask.verbose()
        self.model.net = self.mask.net


    def verbose(self):
        print()
        print('Train Loss................: {:.2f}'.format(self.state['train_loss']))
        print('Test Loss.................: {:.2f}'.format(self.state['test_loss']))
        print('Train Accuracy............: {:.2f}'.format(self.state['train_acc']))
        print('Test Accuracy.............: {:.2f}'.format(self.state['test_acc']))
        print()
        print('Current Learning Rate.....: {:.10f}'.format(self.model.optimizer.param_groups[0]['lr']))
        print('Best Test Accuracy........: {:.2f}'.format(self.state['best_acc']))


    def to_tensorbard(self, epoch):
        self.writer.add_scalar('Loss/train', self.state['train_loss'], epoch)
        self.writer.add_scalar('Accuracy/train', self.state['train_acc'], epoch)
        self.writer.add_scalar('Loss/test', self.state['test_loss'], epoch)
        self.writer.add_scalar('Accuracy/test', self.state['test_acc'], epoch)
        self.writer.add_scalar('Learning Rate/lr', self.model.optimizer.param_groups[0]['lr'], epoch)


    def update_state(self, epoch):
        train_loss, train_acc = self.train()
        test_loss, test_acc = self.test()
        if self.model.summary['scheduler']=='ROP':
            self.model.scheduler.step(test_loss)
        else:
            self.model.scheduler.step()
        self.state['train_loss'] = train_loss
        self.state['train_acc']  = train_acc
        self.state['test_loss'] = test_loss
        self.state['test_acc'] = test_acc
        if test_acc > self.state['best_acc']:
            self.state['best_acc'] = test_acc


    def one_epoch_step(self, epoch):
        print(80*'_')
        print('EPOCH %d / %d' % (epoch+1, self.config['nb_epochs']))
        self.update_state(epoch)
        self.to_tensorbard(epoch)
        if self.config['verbose']:
            self.verbose()
        if self.use_pruning:
            self.prune(epoch)


    @timed
    def run(self):
        for epoch in range(self.config['nb_epochs']):
            self.one_epoch_step(epoch)
            if self.config['use_early_stopping']:
                self.early_stopping(self.state['test_loss'], self.model.net)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break
        score2019(self.model)
