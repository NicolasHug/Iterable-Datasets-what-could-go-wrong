import torch
import torch.utils.data as data
import torch.distributed as dist

def replace_print():
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        for rank in range(dist.get_world_size()):
            if rank == dist.get_rank():
                builtin_print(f"[GPU {rank}]", *args, **kwargs)
            dist.barrier()

    __builtin__.print = print



class MyMapStyleDS:
    
    def __init__(self, size=100):
        self.size = size
        
    def __getitem__(self, i):  # Returns the i'th sample
        s = i
        return s
    
    def __len__(self):
        return self.size


class MyIterableDS(data.IterableDataset):
    
    def __init__(self, size=100):
        self.size = size
        
    def __iter__(self):  # iterate over samples
        for s in range(self.size):
            yield s

        # # Need to shard across DDP workers
        # num_ddp_workers = dist.get_world_size()
        # ddp_worker_id = dist.get_rank()
        
        # for i, s in enumerate(range(self.size)):
        #     if i % num_ddp_workers == ddp_worker_id:
        #         yield s

        # # # But that's no enough!!
        # # # Need to shard across DDP workers **and** accross DataLoader workers
        # # worker_info = data.get_worker_info()
        # # num_dl_workers = worker_info.num_workers
        # # dl_worker_id = worker_info.id

        # # num_ddp_workers = dist.get_world_size()
        # # ddp_worker_id = dist.get_rank()
        
        # # for i, s in enumerate(range(self.size)):
        # #     if i % num_ddp_workers == ddp_worker_id:
        # #         if i % num_dl_workers == dl_worker_id:
        # #             yield s
        # # # That's **two** levels of (embedded) sharding!
    
    def __len__(self):
        return self.size



# Setting up DDP - you can ignore this
dist.init_process_group(backend="gloo")
replace_print()
dist.barrier()


# Map-style dataset
# ds = MyMapStyleDS()
# sampler = data.DistributedSampler(ds, shuffle=False)
# dl = torch.utils.data.DataLoader(ds, batch_size=10, num_workers=4, sampler=sampler)

# Indexable dataset
ds = MyIterableDS()
dl = torch.utils.data.DataLoader(ds, batch_size=10, num_workers=4)

for i, batch in enumerate(dl):    
    print(batch)
    