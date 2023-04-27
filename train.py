import process_hits
from datetime import datetime

import MinkowskiEngine as ME

import time, calendar
#from resnet_mink import Model
import pdsp_dataset_mink as pdm

import torch
from torch import nn

import numpy as np
from argparse import ArgumentParser as ap

def get_ms_delta(p1, p2):
  diff = p2 - p1
  return 1000*diff.seconds + diff.microseconds/1000


class Trainer:
  def __init__(self,
               rank=0,
               weights=[],
               schedule=False,
               load=None,
               validate=False,
               batch_size=2,
               flatten_out=False):
    self.rank=rank
    self.weights=weights
    self.schedule=schedule
    self.load=load
    self.validate=validate
    self.batch_size=batch_size
    self.flatten_out=flatten_out
    self.stored_truths=False
    self.stored_val_truths=False

    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

  def setup_trainers(self, model_type=0, lr=1.e-3):
    if model_type == 0:
      import resnet_mink
      self.model = resnet_mink.Model()
    else:
      import uresnet_mink
      self.model = uresnet_mink.Model()

    self.model.to(self.device)
    #  print('Found cuda. Sending to gpu', self.rank)
    #  self.model.to(rank)
    #  print(next(self.model.parameters()).device)
  
    if len(self.weights) == 0:
      print('Not weighting') 
      self.loss_fn = nn.CrossEntropyLoss()
    else:
      print('Weighting', self.weights)
      self.loss_fn = nn.CrossEntropyLoss(
          weight=torch.tensor(self.weights).float().to(self.device))
  
    self.optimizer = torch.optim.SGD(
        self.model.parameters(), lr=lr, momentum=0.9)
  
    '''
    scheduler = (
      torch.optim.lr_scheduler.CyclicLR(optimizer,
                                        base_lr=0.001,
                                        max_lr=0.01) if schedule
      else None)
    '''
  
    '''
    if load:
      print('Loading from', load)
      ckp = torch.load(load)
      net.load_state_dict(ckp['model_state_dict'])
      optimizer.load_state_dict(ckp['optimizer_state_dict'])
      if scheduler: scheduler.load_state_dict(ckp['scheduler_state_dict'])
    '''
  
    #return (net, loss_fn, optimizer, scheduler)
    #return (net, loss_fn, optimizer)

 
  def setup_output(self):
    self.losses = []
    self.preds = []
    self.truths = []

    self.val_losses = []
    self.val_preds = []
    self.val_truths = []
    self.val_accs = []
    
  def train_loop(self, loader, max_iter=-1):
    self.model.train()
    #torch.cuda.empty_cache()
    #device = get_device()
  
    #size = loader.nevents
    size = len(loader) 
  
    self.losses.append([])
    self.preds.append([])
    if not self.flatten_out: self.truths.append([])
    #for batch, (locs, features, y) in enumerate(loader):
    end = datetime.now()
    for batch, data in enumerate(loader):
        #loader.get_training_batches(batch_size=self.batch_size)):
      begin = datetime.now()
      print('iterate:', begin - end)
      if max_iter > 0 and batch >= max_iter: break
      locs, features, y = data['coordinates'], data['features'], data['labels']
      print('loading data', datetime.now() - begin)
  
      #Zero out gradients
      p1 = datetime.now()
      self.optimizer.zero_grad()
      p2 = datetime.now()
      delta_grad = get_ms_delta(p1, p2)
  
      # Compute prediction error
      #pred = self.model(x.float().to(rank))
      #loss = self.loss_fn(pred, y.long().argmax(1).to(rank))
      p1 = datetime.now()
      the_input = ME.SparseTensor(features, locs, device=self.device)
      p2 = datetime.now()
      delta_input = get_ms_delta(p1, p2)


      start = datetime.now()
      pred = self.model(the_input)
      end1 = datetime.now()
      #print(pred.shape, y.shape)
      loss = self.loss_fn(pred.features, y.to(self.device))
      end2 = datetime.now()

      delta1 = get_ms_delta(start, end1)
      delta2 = get_ms_delta(end1, end2)
  
      # Backpropagation
      p1 = datetime.now()
      loss.backward()
      p2 = datetime.now()
      self.optimizer.step()
      p3 = datetime.now()

      delta_backward = get_ms_delta(p1, p2)
      delta_step = get_ms_delta(p2, p3)
  
      # Adjusting learning rate if available
      #if scheduler:
      #  lrs_list[-1].append(scheduler.get_last_lr())
      #  scheduler.step()
  
      p1 = datetime.now()
      loss = loss.item()
      p2 = datetime.now()
      delta_loss_check = get_ms_delta(p1, p2)
      print(f"loss: {loss:>7f}  [{batch:>5d}/{size}] Time: [pred:{delta1}, loss:{delta2}]")
      #print(pred.features.detach().numpy().argmax(1), y)


      p1 = datetime.now()
      if self.flatten_out:
        self.preds[-1] += [i for i in pred.features.cpu().detach().numpy().argmax(1)]
        if not self.stored_truths:
          self.truths += [i for i in y.numpy().argmax(1)]
      else:
        self.preds[-1].append(pred.features.cpu().detach().numpy().argmax(1))
        if not self.stored_truths:
          self.truths[-1].append(y)
      self.losses[-1].append(loss)
      if batch % 10 == 0: torch.cuda.empty_cache()
      p2 = datetime.now()
      delta_save = get_ms_delta(p1, p2)
      print(f"\tzerograd: {delta_grad} backprop: {delta_backward} step: {delta_step} save: {delta_save}")
      print(f"\tloss_check: {delta_loss_check} input: {delta_input}")
      end = datetime.now()
    self.stored_truths = True
    
  def validate_loop(self, loader, max_iter=-1):

    self.model.eval()
    #size = loader.nevents
    size = len(loader) 
    self.val_losses.append([])
    self.val_preds.append([])
    if not self.flatten_out: self.val_truths.append([])
    correct = 0
    
    with torch.no_grad():
      for batch, data in enumerate(loader):
        if max_iter > 0 and batch >= max_iter: break
        locs, features, y = data['coordinates'], data['features'], data['labels']
        # Compute prediction error
  
        the_input = ME.SparseTensor(features, locs, device=self.device)
        pred = self.model(the_input)
        loss = self.loss_fn(pred.features, y.to(self.device))

        #pred = self.model([torch.LongTensor(locs), torch.FloatTensor(features)])
        #loss = self.loss_fn(pred, torch.Tensor(y).argmax(1))
        loss, current = loss.item(), batch
        self.val_losses[-1].append(loss)
        #print(pred.features.detach().numpy().argmax(1), y)
        #print((pred.argmax(1) == y.argmax(1)))
        correct += np.sum(pred.features.cpu().detach().numpy().argmax(1) == y.numpy())
        #print(pred.features.detach().numpy().argmax(1) == y)

        if self.flatten_out:
          self.val_preds[-1] += [i for i in pred.features.cpu().detach().numpy().argmax(1)]
          if not self.stored_val_truths:
            self.val_truths += [i for i in y.numpy().argmax(1)]
        else:
          self.val_preds[-1].append(pred.features.cpu().detach().numpy().argmax(1))
          self.val_truths[-1].append(y)
        #print(f"loss: {loss:>7f}  [{current:>5d}/{size}]")
        if batch % 10 == 0: torch.cuda.empty_cache()
 
  
    correct /= (1.*loader.dataset.pdsp_data.nevents)
    self.val_accs.append(correct)
    print(f'Validation accuracy: {100.*correct}')
    self.stored_val_truths = True

  def train(self, train_data, validate_data=None, epochs=1):
    for e in range(epochs):
      print('Start epoch', e)

      self.train_loop(train_data)
      #if (e % save_every == 0 or e == epochs-1) and rank == 0:
      #  print('Saving at epoch', e)
      #  save_checkpoint(model, optimizer, scheduler, e)

      if validate_data:
        print('Validating')
        self.validate_loop(validate_data)

      #if scheduler:
      #  scheduler.step()

  def pad_output(self, preds):
    print(preds)
    padded_preds = np.zeros((len(preds), len(preds[0]),
  			   np.max([len(p) for p in preds[0]])))
    print(padded_preds.shape)
    for i in range(len(preds)):
      for j in range(len(preds[i])):
        p = preds[i][j]
        padded_preds[i,j,:len(p)] = p
    return padded_preds

  def save_output(self, validate=False, output_dir='.', pad_pred_truths=True):
    import h5py as h5 
    
    with h5.File(f'{output_dir}/pdsp_training_losses_{calendar.timegm(time.gmtime())}.h5', 'a') as h5out:
      h5out.create_dataset('losses', data=np.array(self.losses))

      if pad_pred_truths:
        padded_preds = self.pad_output(self.preds)
        padded_truths = self.pad_output(self.truths)
      #print(type(self.preds[0][0]))
      #print(type(self.truths[0][0]))
      #print(np.array(self.preds).shape, np.array(self.truths).shape)
      h5out.create_dataset('preds',
                           data=np.array(padded_preds if pad_pred_truths else self.preds))
      h5out.create_dataset('truths',
                           data=np.array(padded_truths if pad_pred_truths else self.truths))
      #h5out.create_dataset('lrs', data=np.array(lrs))
      if validate:
        if pad_pred_truths:
          padded_val_preds = self.pad_output(self.val_preds)
          padded_val_truths = self.pad_output(self.val_truths)
          h5out.create_dataset('val_preds', data=np.array(padded_val_preds))
          h5out.create_dataset('val_truths', data=np.array(padded_val_truths))
        h5out.create_dataset('val_losses', data=np.array(self.val_losses))
        h5out.create_dataset('accuracies', data=np.array(self.val_accs))


def get_weights(pdsp_data, args):
  if not args.noweight:
    return []

  if len(args.weights) > 0: return args.weights

  return pdsp_data.get_sample_weights()

if __name__ == '__main__':
  torch.multiprocessing.set_sharing_strategy('file_system') 
  parser = ap()
  parser.add_argument('--trainsample', required=True)
  parser.add_argument('--validatesample', default=None)
  #parser.add_argument('--filters', nargs=3, default=[128, 192, 256], type=int)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--epochs', type=int, default=1)
  parser.add_argument('--output_dir', type=str, default='./')
  parser.add_argument('--noweight', action='store_false')
  parser.add_argument('--weights', nargs=4, default=[], type=float)
  parser.add_argument('--nload', type=int, default=-1)
  args = parser.parse_args()

  pdsp_data = process_hits.PDSPData(linked=True)
  if args.nload > 0:
    pdsp_data.load_h5_mp(args.trainsample, args.nload)
  else:
    pdsp_data.load_h5(args.trainsample)
  pdsp_data.clean_events()

  loader = pdm.get_loader(pdsp_data, args)

  if args.validatesample:
    validate_data = process_hits.PDSPData(linked=True)
    validate_data.load_h5(args.validatesample)
    validate_data.clean_events()
    val_loader = pdm.get_loader(pdsp_data, args)
  else:
    #validate_data = None
    val_loader = None
 
  #weights = (pdsp_data.get_sample_weights() if args.noweight else [])
  weights = get_weights(pdsp_data, args)
  print(weights)

  trainer = Trainer(batch_size=args.batch_size,
                    weights=weights)
  trainer.setup_trainers()
  trainer.setup_output()
  #trainer.train(pdsp_data, validate_data=validate_data, epochs=args.epochs)
  trainer.train(loader, epochs=args.epochs, validate_data=val_loader)
  trainer.save_output((args.validatesample is not None), output_dir=args.output_dir)
