"""
Training script updates for smart hybrid and multi-GPU training.

This file contains the additions needed for train.py:
1. DDP (Distributed Data Parallel) support
2. Differential learning rates (base vs projection)
3. base_model and freeze_base arguments

Add these to train.py:
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os


# ========================================
# ADD THESE ARGUMENT PARSER ADDITIONS
# ========================================

def add_smart_hybrid_arguments(parser):
    """Add arguments for smart hybrid training."""
    
    # Model architecture
    parser.add_argument('--base_model', type=str, default=None,
                       help='Base model to use (default: google/embeddinggemma-300m)')
    parser.add_argument('--freeze_base', action='store_true',
                       help='Freeze base encoder (train projection only)')
    
    # Differential learning rates
    parser.add_argument('--base_learning_rate', type=float, default=None,
                       help='Learning rate for base encoder (default: same as --learning_rate)')
    parser.add_argument('--projection_learning_rate', type=float, default=None,
                       help='Learning rate for projection layer (default: same as --learning_rate)')
    
    # Multi-GPU
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training (set by torchrun)')


# ========================================
# DDP SETUP FUNCTIONS
# ========================================

def setup_ddp():
    """Setup distributed data parallel training."""
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        return True, device, local_rank, world_size, rank
    else:
        # Single GPU or CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return False, device, 0, 1, 0


def cleanup_ddp():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ========================================
# OPTIMIZER SETUP WITH DIFFERENTIAL LR
# ========================================

def create_optimizer_with_diff_lr(model, args):
    """
    Create optimizer with differential learning rates.
    
    If freeze_base=True: only projection params
    If base_lr != proj_lr: use parameter groups
    Otherwise: standard single LR
    """
    # Get parameter groups
    param_groups_dict = model.get_trainable_parameters()
    
    base_lr = args.base_learning_rate if args.base_learning_rate is not None else args.learning_rate
    proj_lr = args.projection_learning_rate if args.projection_learning_rate is not None else args.learning_rate
    
    # Check if we need differential learning rates
    use_diff_lr = (
        base_lr != proj_lr and
        len(param_groups_dict['base']) > 0
    )
    
    if use_diff_lr:
        print(f"Using differential learning rates:")
        print(f"  Base encoder: {base_lr}")
        print(f"  Projection: {proj_lr}")
        
        param_groups = [
            {
                'params': param_groups_dict['base'],
                'lr': base_lr,
                'name': 'base'
            },
            {
                'params': param_groups_dict['projection'],
                'lr': proj_lr,
                'name': 'projection'
            }
        ]
    else:
        # Single learning rate
        param_groups = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.learning_rate if not use_diff_lr else None,  # Will use group LRs if differential
        weight_decay=args.weight_decay
    )
    
    return optimizer


# ========================================
# EXAMPLE USAGE IN main()
# ========================================

def example_main_updates():
    """
    Example of how to update main() function.
    
    Replace sections in train.py with these:
    """
    
    # 1. Setup DDP
    is_distributed, device, local_rank, world_size, rank = setup_ddp()
    is_main_process = rank == 0
    
    # Only log from main process
    if not is_main_process:
        logging.getLogger().setLevel(logging.WARNING)
    
    # 2. Create model with new parameters
    model = IsotropicGaussianEncoder(
        output_dim=args.output_dim,
        base_model=args.base_model,
        freeze_base=args.freeze_base,
        freeze_early_layers=args.freeze_early_layers
    )
    model = model.to(device)
    
    # 3. Wrap with DDP if distributed
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
    
    # 4. Create data loaders with DistributedSampler
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,  # Use sampler instead of shuffle
            collate_fn=collate_fn,
            num_workers=0
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    # 5. Create optimizer with differential LR
    optimizer = create_optimizer_with_diff_lr(model, args)
    
    # 6. In training loop, set epoch for sampler
    for epoch in range(1, args.epochs + 1):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        # ... rest of training loop
    
    # 7. Cleanup at end
    cleanup_ddp()


# ========================================
# SAVING/LOADING WITH DDP
# ========================================

def save_checkpoint_ddp(model, optimizer, scheduler, args, path, is_main_process):
    """Save checkpoint in DDP training (only from main process)."""
    if not is_main_process:
        return
    
    # Unwrap DDP if needed
    model_to_save = model.module if isinstance(model, DDP) else model
    
    torch.save({
        'epoch': args.epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': vars(args)
    }, path)


# ========================================
# USAGE INSTRUCTIONS
# ========================================

"""
To use these updates:

1. Add imports at top of train.py:
   import torch.distributed as dist
   from torch.nn.parallel import DistributedDataParallel as DDP
   from torch.utils.data.distributed import DistributedSampler

2. Add smart hybrid arguments to parser:
   add_smart_hybrid_arguments(parser)

3. In main(), replace model initialization:
   is_distributed, device, local_rank, world_size, rank = setup_ddp()
   model = IsotropicGaussianEncoder(
       output_dim=args.output_dim,
       base_model=args.base_model,
       freeze_base=args.freeze_base,
       freeze_early_layers=args.freeze_early_layers
   )
   if is_distributed:
       model = DDP(model, device_ids=[local_rank])

4. Replace optimizer creation:
   optimizer = create_optimizer_with_diff_lr(model, args)

5. Use distributed sampler if needed

6. Only save/log from main process

Then run with torchrun:
  # Single GPU
  python scripts/train.py [args]
  
  # Multi-GPU (4 GPUs)
  torchrun --nproc_per_node=4 scripts/train.py [args]
  
  # Smart hybrid
  python scripts/train.py \\
    --base_model sentence-transformers/all-mpnet-base-v2 \\
    --freeze_base \\
    --projection_learning_rate 5e-4
"""

