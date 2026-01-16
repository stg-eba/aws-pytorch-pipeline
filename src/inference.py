import torch
import torch.nn as nn
import json
import logging
import os

logger = logging.getLogger(__name__)

# --- 1. Define the Architecture (Must match training!) ---
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

# --- 2. Load the Model (model_fn) ---
def model_fn(model_dir):
    """
    SageMaker calls this function to load the model artifact from disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    model_path = os.path.join(model_dir, 'model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Failed to load model from {model_path}")
        
    logger.info(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# --- 3. Handle Input Data (input_fn) ---
def input_fn(request_body, request_content_type):
    """
    SageMaker calls this to parse the incoming web request.
    We expect a JSON list of 784 numbers (28x28 pixels).
    """
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        tensor = torch.tensor(data, dtype=torch.float32)
        return tensor.view(1, 1, 28, 28)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

# --- 4. Prediction Logic (predict_fn) ---
def predict_fn(input_data, model):
    """
    Passes the parsed input through the loaded model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = input_data.to(device)
    
    with torch.no_grad():
        output = model(input_data)
        prediction = output.argmax(dim=1, keepdim=True).item()
        
    return prediction