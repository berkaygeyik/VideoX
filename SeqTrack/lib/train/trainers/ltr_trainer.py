import os
from collections import OrderedDict
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import lib.utils.misc as misc
import matplotlib.pyplot as plt


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)
        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

        # Loss history for plotting
        self.history = {'loss': {}, 'iou': {}}

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()
        print(self.epoch)
        print(loader.training)

        for i, data in enumerate(loader, 1):
            # print("start")
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            # forward pass
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            torch.cuda.synchronize()

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)


    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()
        print("completed")
        # Plot loss after each epoch
        self.plot_loss()
        self.plot_iou()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)

                # Record loss
                if name == "Loss/total":
                    self.record_values(loader, val.avg, "loss")
                if name == "IoU":
                    self.record_values(loader, val.avg, "iou")   
            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            if misc.is_main_process():
                print(self.settings.log_file)
                with open(self.settings.log_file, 'a') as f:
                    f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
        
    def plot_loss(self):
        # epochs = [6.8604121589660645, 6.658260054588318, 6.44973708152771, 6.367091581821442, 6.323363452911377, 6.310667231877645, 6.270703454698835, 6.24242527961731, 6.205339643690321, 6.17610852432251, 6.16202563979409, 6.1403181052207945, 6.129853204580454, 6.108130395071847, 6.090022206624349, 6.078288774490357, 6.068869955960442, 6.058121303982205, 6.046297592363859, 6.0339844455719, 6.028554473150344, 6.022225131121549, 6.012605551844058, 6.00549214720726, 6.002628485488891, 5.995079770454994, 5.990020482451827, 5.990826178278242, 5.986564808549552, 5.979036819775899, 5.9711181726763325, 5.966976551115513]
        
        loss_values = [loss for loss in self.history['loss'][self.epoch]]
        
        plt.figure(figsize=(10, 5))
        plt.plot(loss_values, label='Loss/Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        os.makedirs(os.path.dirname("./charts/seqtrack-seqtrack_b256/loss"), exist_ok=True)
    
        # Saving the plot as a PNG file to the specified path
        plt.savefig("./charts/seqtrack-seqtrack_b256/loss/loss_train" + str(self.epoch) + ".png")

        plt.close()

    def plot_loss_checkpoint(self):
        # Getting loss values for the last 10 epochs

        loss_values = []
        for i in range(self.epoch - 10, self.epoch):
            loss_values.extend(self.history['loss'][i])
        
        print("loss_values")
        print(loss_values)


        plt.figure(figsize=(10, 5))
        plt.plot(loss_values, label='Loss/Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over the Last Checkpoint')
        plt.legend()
        
        os.makedirs(os.path.dirname("./charts/seqtrack-seqtrack_b256/loss"), exist_ok=True)
        
        # Saving the plot as a PNG file to the specified path
        plt.savefig(f"./charts/seqtrack-seqtrack_b256/loss/loss_train_last_checkpoint.png")
        plt.close()

    def plot_iou(self):
        iou_values = [iou for iou in self.history['iou'][self.epoch]]
    
        # Creating epoch values starting from 50 and incrementing by 50
        epochs = list(range(self.settings.print_interval, self.settings.print_interval * (len(iou_values) + 1), self.settings.print_interval))
        
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, iou_values, label='IoU/Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Iou')
        plt.title('Iou over Epochs')
        plt.legend()
        plt.xticks(epochs)

        os.makedirs(os.path.dirname("./charts/seqtrack-seqtrack_b256/iou"), exist_ok=True)
    
        # Saving the plot as a PNG file to the specified path
        plt.savefig("./charts/seqtrack-seqtrack_b256/iou/iou_train" + str(self.epoch) + ".png")

        plt.close()

    def plot_iou_checkpoint(self):
        # Getting iou values for the last 10 epochs
        iou_values = self.history['iou'][self.epoch][-10:]
        
        plt.figure(figsize=(10, 5))
        plt.plot(iou_values, label='IoU/Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Iou')
        plt.title('Iou over the Last Checkpoint')
        plt.legend()
        
        os.makedirs(os.path.dirname("./charts/seqtrack-seqtrack_b256/iou"), exist_ok=True)
        
        # Saving the plot as a PNG file to the specified path
        plt.savefig(f"./charts/seqtrack-seqtrack_b256/iou/iou_train_last_checkpoint.png")
        plt.close()

    def record_values(self, loader, val, phase):
        if loader.training:
            if self.epoch not in self.history[phase]:
                self.history[phase][self.epoch] = []
            self.history[phase][self.epoch].append(val)
