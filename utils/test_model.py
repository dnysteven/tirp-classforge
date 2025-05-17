import sys
import torch
# make sure Python can find your util module
sys.path.append('C:\Users\hany_\Documents\MDS\Semester 3\TIRP\Group project\tirp-classforge\rl_app')  

from deep_rl_utils import load_model, STATE_DIM, ACTION_DIM

def sanity_check(ckpt_path):
    # 1) Load the network
    model = load_model(STATE_DIM, ACTION_DIM, ckpt_path)
    model.eval()

    # 2) Print its architecture
    print("=== Model Architecture ===")
    print(model)

    # 3) Total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}\n")

    # 4) Forward pass on a random state
    dummy_state = torch.randn(1, STATE_DIM)
    q_vals = model(dummy_state)
    print("=== Sample Q-values ===")
    print("Shape:", q_vals.shape)
    print(q_vals)

if __name__ == "__main__":
    ckpt = "/full/path/to/deep_rl_model.pth"
    sanity_check(ckpt)
