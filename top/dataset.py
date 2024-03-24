from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from . import collate_fn, initialize_datasets

def retrieve_dataloaders(batch_size, num_workers = 4, num_train = -1, num_valid=-1, num_test=-1, datadir = './data'):
    # Initialize dataloader
    datasets = initialize_datasets(datadir, num_pts={'train':num_train, 'valid':num_valid, 'test':num_test})
        
    # distributed training
    train_sampler = DistributedSampler(datasets['train'])
    
    # Construct PyTorch dataloaders from datasets
    collate = lambda data: collate_fn(data, scale=1, add_beams=False, beam_mass=1)
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=batch_size if (split == 'train') else batch_size, # prevent CUDA memory exceeded
                                     sampler=train_sampler if (split == 'train') else DistributedSampler(dataset, shuffle=False),
                                     pin_memory=True,
                                     persistent_workers=True,
                                     drop_last= True if (split == 'train') else False,
                                     num_workers=num_workers,
                                     collate_fn=collate)
                        for split, dataset in datasets.items()}

    return train_sampler, dataloaders
