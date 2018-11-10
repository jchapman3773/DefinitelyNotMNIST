#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image

for i, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']):
        directory = f'../data/large/{letter}/'
        files = os.listdir(directory)
        label = np.array([0]*10)
        label[i] = 1
        for file in files:
            try:
                im = Image.open(directory+file)
            except:
                print("Delete a corrupted file: " + file)
                os.system(f'rm {directory+file}')
                continue
