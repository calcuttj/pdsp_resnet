import numpy as np
from math import ceil, sqrt, exp, pi
import h5py as h5
import multiprocessing as mp

class PlaneData:
  def __init__(self, wire, time, integral, origin):
    self.wire = wire
    self.time = time
    self.integral = integral
    self.origin = origin

class PDSPData:
  def __init__(self, maxtime=913, maxwires=[800, 800, 480]):
    self.maxtime=maxtime
    self.maxwires=maxwires

  def find_duplicates(self, coords):
    u, counts = np.unique(coords, return_counts=True, axis=0)
    duplicate_indices = []
    dups = np.where(counts > 1)
    if len(dups) == 0: return

    for i in np.where(counts > 1)[0]:
      #print(i, u[i])
      #print(np.where(np.all(coords == u[i], axis=1)))

      duplicate_indices.append(np.where(np.all(coords == u[i], axis=1))[0])
    duplicate_indices = np.array(duplicate_indices)
    return duplicate_indices

  def get_plane_data(self, h5in, k, eid, pid):
    ##To-Do: check maxtime and wires
    #events = np.array(h5in[f'{k}/events/event_id'][:])
    hit_events = np.array(h5in[f'{k}/plane_{pid}_hits/event_id'][:])
    #print(hit_events)
    #print(eid)
    indices = np.all(hit_events == eid, axis=1)
    #print(np.array(h5in[f'{k}/plane_{pid}_hits']['wire'])[indices].flatten().astype(int))

    temp_origin = np.array(h5in[f'{k}/plane_{pid}_hits']['origin'])[indices].flatten()
    origins = np.zeros((*temp_origin.shape, 3))
    origins[np.where(temp_origin < 0), 2] = 1.
    origins[np.where(temp_origin >= 0), 0] = 1. - temp_origin[np.where(temp_origin >= 0)]
    origins[np.where(temp_origin >= 0), 1] = temp_origin[np.where(temp_origin >= 0)]

    plane_data = PlaneData(
               np.array(h5in[f'{k}/plane_{pid}_hits']['wire'])[indices].flatten().astype(int),
               (912 - (np.array(h5in[f'{k}/plane_{pid}_hits']['time'])[indices]- 500.)/6.025).flatten().astype(int),
               np.array(h5in[f'{k}/plane_{pid}_hits']['integral'])[indices].flatten(),
               origins 
           )
    coords = np.zeros((len(plane_data.wire), 2))
    coords[:, 0] = plane_data.wire
    coords[:, 1] = plane_data.time
    duplicate_indices = self.find_duplicates(coords)
    #print(self.duplicate_indices)

    to_del = []
    for i in range(len(duplicate_indices)):
      #print(i, self.duplicate_indices[i])
      #print(plane_data.integral[self.duplicate_indices[i]])
      #print(plane_data.origin[self.duplicate_indices[i]])
      dups = duplicate_indices[i]
      dups.sort()

      ##Weight the origins by the integral size
      plane_data.origin[dups] *= plane_data.integral[dups].reshape(*plane_data.integral[dups].shape, 1)
      plane_data.integral[dups[0]] = np.sum(plane_data.integral[dups])
      plane_data.origin[dups[0]] = np.sum(plane_data.origin[dups], axis=0)
      plane_data.origin[dups[0]] /= plane_data.integral[dups[0]]
      #print(plane_data.integral[dups[0]])
      #print(plane_data.origin[dups[0]])
      to_del += [i for i in dups[1:]]
    #to_del = np.array(to_del).flatten()
    #print(k, eid, to_del)
    to_del.sort()
    #print(to_del)

    

    #to_del = np.where((plane_data.time >= self.maxtime) |
    #                  (plane_data.wire >= self.maxwires[pid]))

    #print('Wire', pid, plane_data.wire)
    #print('Time', pid, plane_data.time)
    #print('ToD', pid, to_del)
    if len(to_del) > 0:
      plane_data.time = np.delete(plane_data.time, to_del)
      plane_data.wire = np.delete(plane_data.wire, to_del)
      plane_data.integral = np.delete(plane_data.integral, to_del)
      plane_data.origin = np.delete(plane_data.origin, to_del, axis=0)
      #print(plane_data.time.shape, plane_data.origin.shape)
    return plane_data



  def load_file_mp(self, h5in, procid):

    #start = self.split_keys[procid].index('link1557')
    #start = 0
    #for a, k in enumerate(self.split_keys[procid][start:]):
    for a, k in enumerate(self.split_keys[procid]):

      events = [i for i in np.array(h5in[f'{k}/events/event_id'])]

      if not a % 100:
        with self.lock:
          string = ''
          self.split_count[procid] = a
          for i in range(len(self.split_count)):
            string += f'{self.split_count[i]}/{len(self.split_keys[i])} '
          #print(f'{procid}: {a}/{len(self.split_keys[procid])}'), end='\r')
          print(string, end='\r')
          #print(string, end='\x1b[1K\r')

      all_plane_datas = []
      for eid in events:
        all_plane_datas.append(self.get_plane_data(h5in, k, eid, 2))
      #nhits = [i for i in np.array(h5in[f'{k}/events/nhits'][:])]
      nhits = np.array(h5in[f'{k}/events/nhits'])
      #print('nhits:', nhits)

      with self.lock:
        #print(len(events), len(all_plane_datas))
        for i in range(len(events)):
          #self.nhits.append(nhits[i])
          self.events.append(events[i])
        for plane_data in all_plane_datas:
          self.plane2_time.append(plane_data.time)
          self.nhits.append([0, 0, len(plane_data.time)])
          self.plane2_wire.append(plane_data.wire)
          self.plane2_integral.append(plane_data.integral)
          self.plane2_origin.append(plane_data.origin)

  def load_h5_mp(self, filename, num_workers):
    with h5.File(filename, 'r') as h5in:
      self.loaded_truth = False

      ##Make lock and manager
      ##And associated lists
      with mp.Manager() as manager:
        self.lock = mp.Lock()
        self.events = manager.list() 
        self.plane0_time = manager.list()
        self.plane0_wire = manager.list()
        self.plane0_integral = manager.list()
        self.plane0_origin = manager.list()

        self.plane1_time = manager.list()
        self.plane1_wire = manager.list()
        self.plane1_integral = manager.list()
        self.plane1_origin = manager.list()

        self.plane2_time = manager.list()
        self.plane2_wire = manager.list()
        self.plane2_integral = manager.list()
        self.plane2_origin = manager.list()

        self.keys = [k for k in h5in.keys()]
        self.split_keys = manager.list([
          self.keys[i::num_workers]
          for i in range(num_workers)
        ])

        self.split_count = manager.list([0 for i in range(num_workers)])

        self.nhits = manager.list() 

        procs = [
          mp.Process(target=self.load_file_mp,
                     args=(h5in, i))
          for i in range(num_workers)
        ]

        for p in procs:
          p.start()
        for p in procs:
          p.join()

        self.plane0_time = list(self.plane0_time)
        self.plane0_wire = list(self.plane0_wire)
        self.plane0_integral = list(self.plane0_integral)
        self.plane0_origin = list(self.plane0_origin)

        self.plane1_time = list(self.plane1_time) 
        self.plane1_wire = list(self.plane1_wire)
        self.plane1_integral = list(self.plane1_integral)
        self.plane1_origin = list(self.plane1_origin)

        self.plane2_time = list(self.plane2_time) 
        self.plane2_wire = list(self.plane2_wire)
        self.plane2_integral = list(self.plane2_integral)
        self.plane2_origin = list(self.plane2_origin)

        self.wires = [self.plane0_wire, self.plane1_wire, self.plane2_wire]
        self.times = [self.plane0_time, self.plane1_time, self.plane2_time]
        self.integrals = [self.plane0_integral, self.plane1_integral, self.plane2_integral]
        self.origins = [self.plane0_origin, self.plane1_origin, self.plane2_origin]
        self.nhits = np.array(self.nhits)

        self.nevents = len(self.nhits)
        self.events = np.array(self.events)


  def get_plane(self, eventindex, pid):
    #check eventindex
    if pid not in [0, 1, 2]:
      ##TODO -- throw exception
      return 0

    #nhits = self.nhits[eventindex, pid]
    locations = np.zeros((len(self.wires[pid][eventindex]), 2), dtype=int)
    locations[:, 0] = self.wires[pid][eventindex]
    locations[:, 1] = self.times[pid][eventindex]
    #locations[:, 2] = np.arange(nhits)

    features = np.array([[i] for i in self.integrals[pid][eventindex]])
    origins = np.array([i for i in self.origins[pid][eventindex]])
    return (locations, features), origins


    

  '''
  def make_plane(self, pid, pad2=False, use_width=False):
    plane = np.zeros((self.maxtime, 480 if (pid == 2 and not pad2) else 800))

    if pid not in [0, 1, 2]:
      ##TODO -- throw exception
      return 0
    elif pid == 0: data = self.plane0_data
    elif pid == 1: data = self.plane1_data
    elif pid == 2: data = self.plane2_data


    bin_width=6.025
    for d in data:
      w = int(d['wire'])
      if pid == 2:
        if w > 479: continue
        if pad2: w = w + (800 - 480)//2
      elif w > 799: continue
      t = 912 - int((d['time'] - 500)/bin_width)
      if t >= self.maxtime: continue
      i = d['integral']

      if not use_width:
        plane[t,w] += i
      else:
        rms = d['rms']
        #Get the number of bins away -- 5 sigma
        nbins = ceil(5*rms/bin_width)
        for b in range(t - nbins, t + nbins + 1):
          if b >= self.maxtime: continue
          plane[b,w] += i*(bin_width/(rms*sqrt(2.*pi)))*exp(-.5*((t - b)*bin_width/rms)**2)
  
    return plane
    '''

  def draw_plane(self, eid, pid=2):
    import matplotlib.pyplot as plt
    plane = self.get_plane(eid, pid)
    coords = plane[0][0]
    integrals = plane[0][1]

    maxtime = np.max(coords[:,1])
    print(maxtime)
    print(np.max(coords[:,0]))
    plot_plane = np.zeros(((480 if pid == 2 else 800), maxtime+1))
    for i in range(len(coords)):
      plot_plane[coords[i][0], coords[i][1]] = integrals[i]
    plt.imshow(plot_plane, cmap='jet')
    plt.show()


  def clean_events(self, check_nhits=True):
  ##Check this -- not working
    pass
    #nohits = np.any(self.nhits == 0, axis=1)
    #print(len(nohits))
    #indices = np.where(
    #  (nohits if check_nhits else check_nhits)
    #)

    ##self.pdg = np.delete(self.pdg, indices)
    ##self.topos = np.delete(self.topos, indices)
    ##self.interacted = np.delete(self.interacted, indices)
    ##self.n_neutron = np.delete(self.n_neutron, indices)
    ##self.n_proton = np.delete(self.n_proton, indices)
    ##self.n_piplus = np.delete(self.n_piplus, indices)
    ##self.n_piminus = np.delete(self.n_piminus, indices)
    ##self.n_pi0 = np.delete(self.n_pi0, indices)
    #self.events = np.delete(self.events, indices)

    ##self.events = np.delete(self.events, indices, axis=0)
    #self.nhits = np.delete(self.nhits, indices, axis=0)
    ##self.event_keys = np.delete(self.event_keys, indices)
    #self.nevents -= len(indices[0])

    #for i in indices[0][::-1]:
    #  #print('Deleting', i)
    #  self.delete_event(i)

  def delete_event(self, i):
    #del self.plane0_time[i]
    #del self.plane0_wire[i]
    #del self.plane0_integral[i]

    #del self.plane1_time[i]
    #del self.plane1_wire[i]
    #del self.plane1_integral[i]

    del self.plane2_time[i]
    del self.plane2_wire[i]
    del self.plane2_integral[i]
    del self.plane2_origin[i]

  def get_nbatches(self, batchsize=2):
    return ceil(self.nevents/batchsize)

  def get_sample_weights(self, pid=2):
    total_nhits = len([origins for origins in self.origins[pid]])
    total_origins = np.sum(np.array([np.sum(origins, axis=0) for origins in self.origins[pid]]), axis=0)
    return total_nhits/total_origins

  def get_training_batches(self, batchsize=2, maxbatches=-1, startbatch=0):

    for i in range(startbatch, ceil(self.nevents/batchsize)):
      if maxbatches > 0 and i > maxbatches: break
      print('Batch', i)
      #print('\t', np.arange(self.nevents)[i*batchsize:(i+1)*batchsize])

      batch_events = np.arange(self.nevents)[i*batchsize:(i+1)*batchsize]
      nb = len(batch_events)
      #plane_batch = np.zeros((nb, 1, ))
      nhits = int(sum(self.nhits[i*batchsize:(i+1)*batchsize, 2]))
      locs_batch = np.zeros((nhits, 3), dtype=int)
      features_batch = np.zeros((nhits, 1))

      truth_batch = np.zeros((nb, 4))
      start = 0
      for a, j in enumerate(batch_events):
        #print(j)
        locations, features = self.get_plane(j, 2)
        locs_batch[start:start+len(locations), :2] = locations
        locs_batch[start:start+len(locations), 2] = a
        features_batch[start:start+len(locations)] = features
        start += len(locations)

        truth_batch[a, self.topos[j]] = 1.

      yield ((locs_batch, features_batch), truth_batch)
