import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state['state_dict'], filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")
