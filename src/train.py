import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# --- HELPER: Fix Folder Structure for Torchvision ---
def organize_data_for_torchvision(root_dir):
    """
    SageMaker puts files in root_dir. 
    Torchvision expects them in root_dir/MNIST/raw.
    We move them there to make Torchvision happy.
    """
    print(f"📂 Organizing data in {root_dir}...")
    
    # Define the structure torchvision expects
    raw_dir = os.path.join(root_dir, 'MNIST', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    # List of expected MNIST files
    files = [
        'train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'
    ]
    
    # Move files from root to root/MNIST/raw
    for f in files:
        src = os.path.join(root_dir, f)
        dst = os.path.join(raw_dir, f)
        
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"   Moved {f} -> {dst}")
        elif os.path.exists(dst):
            print(f"   {f} is already in the right place.")
        else:
            # It's okay if not all exist (sometimes we only upload train)
            pass

# --- 1. Define the Neural Network ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) 
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    
    # We ignore the --model-dir passed by SageMaker and use the hardcoded one below
    parser.add_argument('--model-dir', type=str, default='')
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data'))
    
    args = parser.parse_args()

    # --- CRITICAL FIX: FORCE THE SAVE PATH ---
    # SageMaker expects models to be here to upload them to S3 automatically
    target_model_dir = '/opt/ml/model'
    
    # Fix Data Structure
    organize_data_for_torchvision(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transform & Load
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print(f"Loading data from: {args.data_dir}")
    train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Train
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
    
    train(model, device, train_loader, optimizer, args.epochs)

    # --- SAVE ---
    # Ensure the directory exists
    os.makedirs(target_model_dir, exist_ok=True)
    
    save_path = os.path.join(target_model_dir, "model.pth")
    torch.save(model.state_dict(), save_path)
    
    print(f"✅ Model saved to: {save_path}")
    print(f"📂 Directory contents of {target_model_dir}:")
    print(os.listdir(target_model_dir))

if __name__ == '__main__':
    main()