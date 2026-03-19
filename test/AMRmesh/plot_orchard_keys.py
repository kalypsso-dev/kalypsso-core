# -*- coding: utf-8 -*-

"""
Just plotting Morton orchard keys before / after AMR cycle for illustrative purpose.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex=True)
import os
import fnmatch

# count the number of file which name matches "orchard_keys_before_amr_mpi_0000*.npy"
mpi_size = len(fnmatch.filter(os.listdir('./'), 'orchard_keys_before*.npy'))

# create empty list of array (1 per MPI rank)
list_keys_before = []
list_index_before = []
list_keys_after = []
list_index_after = []

for i in range(mpi_size):
    print(i)

    mpi_rank = "{:05d}".format(i)

    keys_before = np.load("./orchard_keys_before_amr_mpi_"+mpi_rank+".npy")
    index_before = np.arange(keys_before.size)

    keys_after = np.load("./orchard_keys_after_amr_mpi_"+mpi_rank+".npy")
    index_after = np.arange(keys_after.size)

    list_keys_before.append(keys_before)
    list_index_before.append(index_before)
    list_keys_after.append(keys_after)
    list_index_after.append(index_after)

    if i>0:
        for j in range(i):
            list_index_before[i] += list_index_before[j].size
            list_index_after[i] += list_index_after[j].size

    plt.plot(list_index_before[i], list_keys_before[i], label='orchard_keys_before_amr_mpi', color='red')
    plt.plot(list_index_after[i], list_keys_after[i], label='orchard_keys_after_amr_mpi', color='blue')

plt.grid(True)
plt.title('Orchard keys along the Morton curve before / after AMR cycle')
plt.xlabel('key index')
plt.ylabel('Morton - orchard key')

plt.show()
