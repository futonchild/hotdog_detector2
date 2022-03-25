import os
import random
from shutil import copyfile

# Use this to split your HOTDOG and NOTDOG image directories into, e.g., training/hotdog, training/notdog, validation/hotdog, and validation/notdog directories

def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):
    all_fnames = os.listdir( SOURCE )
    num_fnames = len(all_fnames)
    num_train_fnames = round(SPLIT_SIZE * num_fnames)
    num_test_fnames = num_fnames - num_train_fnames
    train_fnames = random.sample(all_fnames, num_train_fnames)
    test_fnames = [(yield fname) for fname in all_fnames if fname not in train_fnames]
    for fname in train_fnames:
        src_fname = os.path.join(SOURCE, fname)
        target_fname = os.path.join(TRAINING, fname)
        if os.path.getsize(src_fname) > 0:
            copyfile(src_fname, target_fname)
    for fname in test_fnames:
        src_fname = os.path.join(SOURCE, fname)
        target_fname = os.path.join(VALIDATION, fname)
        if os.path.getsize(src_fname) > 0:
            copyfile(src_fname, target_fname)

# E.g.:
#split_size = .9
#split_data(HOTDOG_SOURCE_DIR, HOTDOG_TRAINING_DIR, HOTDOG_VALIDATION_DIR, split_size)
#split_data(NOTDOG_SOURCE_DIR, NOTDOG_TRAINING_DIR, NOTDOG_VALIDATION_DIR, split_size)
