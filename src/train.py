#!/usr/bin/env python
import os
from datetime import datetime

import tensorflow as tf

from config import configs
from data_loader import get_loader
from EVFlowNet import EVFlowNet

def main():
    args = configs()
    if args.training_instance:
        args.load_path = os.path.join(args.load_path, args.training_instance)
        args.summary_path = os.path.join(args.summary_path, args.training_instance)
    else:
        args.load_path = os.path.join(args.load_path,
                                      "evflownet_{}".format(datetime.now()
                                                            .strftime("%m%d_%H%M%S")))
        args.summary_path = os.path.join(args.summary_path,
                                         "evflownet_{}".format(datetime.now()
                                                               .strftime("%m%d_%H%M%S")))
    if not os.path.exists(args.load_path):
        os.makedirs(args.load_path)
    if not os.path.exists(args.summary_path):
        os.makedirs(args.summary_path)

    # Fix the random seed for reproducibility.
    # Remove this if you are using this code for something else!
    tf.random.set_seed(12345)
        
    dataset, n_ima = get_loader(
        args.data_path, args.batch_size, args.image_width, args.image_height,
        split='train',
        sequence=args.test_sequence,
        shuffle=True)
    
    print("Number of images: {}".format(n_ima))
    trainer = EVFlowNet(args)
    trainer.train_model(dataset, n_ima)
    
if __name__ == "__main__":
    main()
