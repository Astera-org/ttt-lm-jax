import os
import os.path as osp
import wandb
from ttt.infra.jax_utils import master_print

def log_perplexity_to_wandb(exp_dir, exp_name, perplexity_results, step=None):
    """
    Resume W&B run and log perplexity results.
    
    Args:
        exp_dir: Experiment directory
        exp_name: Experiment name
        perplexity_results: Dictionary or single value of perplexity results
        step: Optional step number for logging
    """
    wandb_id_file = osp.join(exp_dir, exp_name, "wandb_run_id.txt")
    if not osp.exists(wandb_id_file):
        master_print("W&B run ID file not found, cannot log perplexity results")
        return False
    
    try:
        with open(wandb_id_file, 'r') as f:
            wandb_run_id = f.read().strip()
        
        # Resume the existing W&B run
        run = wandb.init(project="TTT-LM", id=wandb_run_id, resume="must")
        master_print(f"Resumed W&B run: {wandb_run_id}")
        
        # Prepare logging dictionary
        log_dict = {}
        
        if isinstance(perplexity_results, dict):
            # If results contain multiple metrics
            for key, value in perplexity_results.items():
                # Clean up key names and add Final_ prefix
                clean_key = key.replace('_', ' ').title()
                if 'ppl' in key.lower() or 'perplexity' in key.lower():
                    log_dict[f"Final_Perplexity_{clean_key}"] = float(value)
                else:
                    log_dict[f"Final_{clean_key}"] = float(value)
        else:
            # If results is a single perplexity value
            log_dict["Final_Perplexity"] = float(perplexity_results)
        
        # Log the results
        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)
        
        master_print(f"Perplexity results logged to W&B: {log_dict}")
        wandb.finish()
        return True
        
    except Exception as e:
        master_print(f"Failed to log perplexity to W&B: {str(e)}")
        return False

def save_wandb_run_id(exp_dir, exp_name, run_id):
    """Save W&B run ID to file for later use."""
    wandb_id_file = osp.join(exp_dir, exp_name, "wandb_run_id.txt")
    os.makedirs(osp.dirname(wandb_id_file), exist_ok=True)
    with open(wandb_id_file, 'w') as f:
        f.write(run_id)
    master_print(f"W&B run ID saved: {run_id}")
