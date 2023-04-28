import MinkowskiEngine as ME
import datetime

import time, calendar
#from resnet_mink import Model

import torch
from torch import nn

import numpy as np
from argparse import ArgumentParser as ap

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier
import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
  def __init__(self,
               rank=0,
               weights=[],
               schedule=False,
               load=None,
               validate=False,
               batch_size=2,
               flatten_out=False,
               noddp=False):
    self.rank=rank
    self.weights=weights
    self.schedule=schedule
    self.load=load
    self.validate=validate
    self.batch_size=batch_size
    self.flatten_out=flatten_out
    self.stored_truths=False
    self.stored_val_truths=False
    self.noddp=noddp

    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

  def setup_trainers(self, model_type=0, lr=1.e-3):
    if model_type == 0:
      import resnet_mink
      model = resnet_mink.Model(dropout=True)
    else:
      import uresnet_mink
      model = uresnet_mink.Model()

    print('Found cuda. Sending to gpu', self.rank)
    torch.cuda.set_device(self.rank)
    model.cuda(self.rank)
    #model.to(self.rank)
    print(next(model.parameters()).device)
    #print(model)
  
    if len(self.weights) == 0:
      print('Not weighting') 
      self.loss_fn = nn.CrossEntropyLoss()
    else:
      print('Weighting', self.weights)
      self.loss_fn = nn.CrossEntropyLoss(
          weight=torch.tensor(self.weights).float().to(self.rank))
  
    self.optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9)
    self.model = model if self.noddp else DDP(model, device_ids=[self.rank])
  
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
    self.val_locs = []
    self.val_nhits = []
    self.val_accs = []
    
  def train_loop(self, loader, max_iter=-1):
    self.model.train()
  
    #size = loader.nevents
    size = len(loader) 
  
    self.losses.append([])
    self.preds.append([])
    if not self.flatten_out: self.truths.append([])

    for batch, data in enumerate(loader):
      if max_iter > 0 and batch >= max_iter: break
      locs, features, y = data['coordinates'], data['features'], data['labels']
  
      #Zero out gradients
      self.optimizer.zero_grad()
  
      # Compute prediction error
      #pred = self.model(x.float().to(rank))
      #loss = self.loss_fn(pred, y.long().argmax(1).to(rank))
      the_input = ME.SparseTensor(features, locs, device=self.rank) ##TODO -- noddp need this?
      pred = self.model(the_input)
      #print(pred.shape, y.shape)
      loss = self.loss_fn(pred.features, y.to(self.rank))
  
      # Backpropagation
      loss.backward()
      self.optimizer.step()
  
      # Adjusting learning rate if available
      #if scheduler:
      #  lrs_list[-1].append(scheduler.get_last_lr())
      #  scheduler.step()
  
      loss, current = loss.item(), batch
      print(f"loss: {loss:>7f}  [{current:>5d}/{size}]")
      #print(pred.features.detach().numpy().argmax(1), y)
      if self.flatten_out:
        self.preds[-1] += [i for i in pred.features.cpu().detach().numpy()]
        if not self.stored_truths:
          self.truths += [i for i in y.numpy()]
      else:
        self.preds[-1].append(pred.features.cpu().detach().numpy())
        if not self.stored_truths:
          self.truths[-1].append(y)
      self.losses[-1].append(loss)
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

        the_input = ME.SparseTensor(features, locs, device=self.rank)
        #print(the_input.device)
        if self.noddp:
          pred = self.model(the_input)
        else:
          pred = self.model.module(the_input)
        loss = self.loss_fn(pred.features, y.to(self.rank))

        #pred = self.model([torch.LongTensor(locs), torch.FloatTensor(features)])
        #loss = self.loss_fn(pred, torch.Tensor(y).argmax(1))
        loss, current = loss.item(), batch
        self.val_losses[-1].append(loss)
        #print(pred.features.detach().numpy().argmax(1), y)
        #print((pred.argmax(1) == y.argmax(1)))
        correct += np.sum(pred.features.cpu().detach().numpy().argmax(1) == y.numpy())
        #print(pred.features.detach().numpy().argmax(1) == y)

        if self.flatten_out:
          self.val_preds[-1] += [i for i in pred.features.cpu().detach().numpy()]
          if not self.stored_val_truths:
            self.val_truths += [i for i in y.numpy()]
        else:
          self.val_preds[-1].append(pred.features.cpu().detach().numpy())
          self.val_truths[-1].append(y)
        if not self.stored_val_truths:
          self.val_locs += [i for i in locs.numpy()]
          indices = set([i for i in locs.numpy()[:, 0]])
          self.val_nhits += [len(np.where(locs.numpy()[:, 0] == i)[0]) for i in indices]
        print(f"loss: {loss:>7f}  [{current:>5d}/{size}]")
  
    correct /= (1.*loader.dataset.pdsp_data.nevents)
    self.val_accs.append(correct)
    print(f'Validation accuracy: {100.*correct}')
    self.stored_val_truths = True

  def barrier(self):
    if self.noddp: return

    print(f'GPU {self.rank} hit barrier')
    barrier()
    print(f'GPU {self.rank} passed barrier')

  def train(self, train_data, validate_data=None, epochs=1):
    for e in range(epochs):
      print('Start epoch', e)

      self.train_loop(train_data)
      #if (e % save_every == 0 or e == epochs-1) and rank == 0:
      #  print('Saving at epoch', e)
      #  save_checkpoint(model, optimizer, scheduler, e)

      if validate_data and self.rank == 0:
        print('Validating')
        self.validate_loop(validate_data)

      self.barrier()

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
        else:
          h5out.create_dataset('val_preds', data=np.array(self.val_preds))
          h5out.create_dataset('val_truths', data=np.array(self.val_truths))
        h5out.create_dataset('val_losses', data=np.array(self.val_losses))
        h5out.create_dataset('accuracies', data=np.array(self.val_accs))
        h5out.create_dataset('val_locs', data=np.array(self.val_locs))
        h5out.create_dataset('val_nhits', data=np.array(self.val_nhits))


def get_weights(pdsp_data, args):
  if not args.noweight:
    return []

  if len(args.weights) > 0: return args.weights

  return pdsp_data.get_sample_weights()

def train(rank: int,
          args,
          weights,
          world_size,
          train_dataset,
          val_dataset=None,
          ):


  if not args.noddp: ddp_setup(rank, world_size)


  if args.type == 0:
    import pdsp_dataset_mink as pdm
  else:
    import pdsp_dataset_mink_allhits as pdm

  loader = pdm.DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=0,
      collate_fn=pdm.minkowski_collate_fn,
      sampler=(None if args.noddp else DistributedSampler(train_dataset)),
  )
  if val_dataset is None:
    val_loader = None
  else:
    val_loader = pdm.DataLoader(
      val_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      collate_fn=pdm.minkowski_collate_fn,
    )

  trainer = Trainer(
      rank=rank,
      batch_size=args.batch_size,
      weights=weights,
      flatten_out=True,
      noddp=args.noddp,
  )

  trainer.setup_trainers(model_type=args.type, lr=args.lr)
  trainer.setup_output()
  trainer.train(loader, epochs=args.epochs, validate_data=val_loader)
  if rank == 0:
    trainer.save_output((args.validatesample is not None),
                        output_dir=args.output_dir,
                        pad_pred_truths=False)
  if not args.noddp: destroy_process_group()

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
  parser.add_argument('--nload', type=int, default=1)
  parser.add_argument('--lr', type=float, default=1.e-3)
  parser.add_argument('--type', type=int, default=0,
                      help=('Model/data type:\n'
                            '0 (Default) -- Event Classification\n'
                            '1 -- Beam/Cosmic Hits'))
  parser.add_argument('--noddp', action='store_true')
  args = parser.parse_args()

  if args.type == 0:
    import pdsp_dataset_mink as pdm
    import process_hits
    pdsp_data = process_hits.PDSPData()
  else:
    import pdsp_dataset_mink_allhits as pdm
    import process_all_hits as process_hits
    pdsp_data = process_hits.PDSPData(nfeatures=2)


  pdsp_data.load_h5_mp(args.trainsample, args.nload)
  pdsp_data.clean_events()

  pdsp_dataset = pdm.get_dataset(pdsp_data)

  if args.validatesample:
    if args.type == 0:
      validate_data = process_hits.PDSPData()
    else:
      validate_data = process_hits.PDSPData(nfeatures=2)

    validate_data.load_h5_mp(args.validatesample, args.nload)
    validate_data.clean_events()
    val_dataset = pdm.get_dataset(validate_data)
  else:
    val_dataset = None
 
  weights = get_weights(pdsp_data, args)
  print(weights)

  if not args.noddp:
    world_size = torch.cuda.device_count() if torch.cuda.is_available else 1
    mp.spawn(train,
      args=(
        args,
        weights,
        world_size,
        pdsp_dataset,
        val_dataset,
      ), nprocs=world_size
    )
  else:
   train(0, args, weights, 1, pdsp_dataset, val_dataset) 
