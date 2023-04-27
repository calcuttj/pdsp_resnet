import train
import torch
from argparse import ArgumentParser as ap

import pdsp_dataset_mink_allhits as pdm


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

  args = parser.parse_args()

  import process_all_hits
  pdsp_data = process_all_hits.PDSPData(nfeatures=2)
  pdsp_data.load_h5_mp(args.trainsample, args.nload)

  loader = pdm.get_loader(pdsp_data, args)

  if args.validatesample:
    validate_data = process_all_hits.PDSPData(nfeatures=2)
    validate_data.load_h5_mp(args.validatesample, args.nload)
    val_loader = pdm.get_loader(validate_data, args)
  else:
    #validate_data = None
    val_loader = None
 
  #weights = (pdsp_data.get_sample_weights() if args.noweight else [])
  weights = train.get_weights(pdsp_data, args)
  print(weights)

  trainer = train.Trainer(batch_size=args.batch_size,
                          weights=weights,
                          flatten_out=True)
  trainer.setup_trainers(model_type=1, lr=1.e-2)
  trainer.setup_output()
  #trainer.train(pdsp_data, validate_data=validate_data, epochs=args.epochs)
  trainer.train(loader, epochs=args.epochs, validate_data=val_loader)
  trainer.save_output((args.validatesample is not None),
                      output_dir=args.output_dir,
                      pad_pred_truths=False)
