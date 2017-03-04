import numpy as np

from segmentation.mPb.training.trainer.AngledmPbTrainer import AngledmPbTrainer

angles = [int(np.rad2deg(np.pi * i / 8)) for i in range(0, 8)]
for angle in angles:
    trainer = AngledmPbTrainer(0)
    trainer.prep()
