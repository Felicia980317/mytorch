import queue
import random
from random import shuffle

import numpy as np
import torch
import torch.multiprocessing as mp
import time


class ParaDataLoader(object):
    def __init__(
        self,
        dataset,
        batch_size,
        prefetch_size=2,
        use_shuffle=True,
        n_workers=1,
        autostart=True,
    ):
        super(ParaDataLoader, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_shuffle = use_shuffle
        self.n_workers = n_workers
        self.n_samples = len(self.dataset)
        self.step_per_epoch = self.n_samples // self.batch_size

        manager = mp.Manager()
        self.q = manager.Queue(maxsize=prefetch_size)
        self.q_pools = manager.Queue(maxsize=prefetch_size)
        self.workers_status = manager.Array("i", [0] * self.n_workers)
        self.is_stop = manager.Value("i", 0)

        if autostart:
            self.start()

    def updateParallel(self, n_workers, workers_status, is_stop):
        try:
            while is_stop.get() == 0:
                if np.sum(workers_status[:]) == n_workers:
                    for i in range(n_workers):
                        workers_status[i] = 0
                time.sleep(0.5)
            print("updateParallel Done.")
        except Exception as e:
            print("Exception in updateParallel:{}".format(e))

    def updatePools(self, batch_size, n_samples, use_shuffle, q_pools, is_stop):
        try:
            random_pools = np.array(range(n_samples))
            if use_shuffle:
                shuffle(random_pools)
            idx = 0
            while is_stop.get() == 0:
                if not q_pools.full():
                    if idx + batch_size > n_samples:
                        idx = 0
                        if use_shuffle:
                            shuffle(random_pools)
                    idx_batch_list = []
                    for idx_batch in random_pools[idx : idx + batch_size]:
                        idx_batch_list.append(idx_batch)
                    idx += batch_size
                    q_pools.put(idx_batch_list)
                time.sleep(0.5)
            print("updatePools Done.")
        except Exception as e:
            print("Exception in updatePools:{}".format(e))

    def run(self, dataset, q, q_pools, worker, workers_status, is_stop):
        try:
            while is_stop.get() == 0:
                if (not q.full()) and (workers_status[worker] == 0):
                    idx_batch_list = q_pools.get(timeout=60)
                    data = []
                    for idx_batch in idx_batch_list:
                        data.append(dataset[idx_batch])
                    out = []
                    for key in data[0].keys():
                        out.append([_data[key] for _data in data])

                    q.put(tuple(torch.stack(_out) for _out in out))
                    workers_status[worker] = 1
                time.sleep(0.25)
            print("run{} Done.".format(worker))

        except Exception as e:
            print("Exception in run:{}".format(e))

    def start(self):
        self.is_stop.set(0)
        p = mp.Process(
            target=self.updatePools,
            args=(
                self.batch_size,
                self.n_samples,
                self.use_shuffle,
                self.q_pools,
                self.is_stop,
            ),
        )
        p.start()

        p = mp.Process(
            target=self.updateParallel,
            args=(self.n_workers, self.workers_status, self.is_stop),
        )
        p.start()

        for i in range(self.n_workers):
            p = mp.Process(
                target=self.run,
                args=(
                    self.dataset,
                    self.q,
                    self.q_pools,
                    i,
                    self.workers_status,
                    self.is_stop,
                ),
            )
            p.start()

    def stop(self):
        self.is_stop.set(1)

    def status(self):
        return self.is_stop.get()

    def dequeue(self):
        return self.q.get(timeout=60)

    def qsize(self):
        return self.q.qsize()

    def __len__(self):
        return self.step_per_epoch

    def __iter__(self):
        return self

    def __next__(self):
        return self.dequeue()


if __name__ == "__main__":
    pass
