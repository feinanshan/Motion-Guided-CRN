import numpy as np
import os
from datetime import datetime
import logging
import cv2


# save log
def save_log(prefix, output_dir):
    fmt = '%(asctime)s %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        datefmt=date_fmt,
                        filename=os.path.join(output_dir,
                                              prefix + '_' + datetime.now().strftime('%Y_%m_%d_%H:%M:%S') + '.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)








