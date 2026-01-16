import boto3
import os
import tarfile
import torch
import sys
from dotenv import load_dotenv

# Setup paths to find your Net class
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from train import Net
except ImportError:
    sys.path.append(os.getcwd())
    from train import Net

load_dotenv()
BUCKET_NAME = os.getenv('BUCKET_NAME')

def get_latest_job_model():
    s3 = boto3.client('s3')
    
    print(f"🔍 Scanning s3://{BUCKET_NAME}/models/ for the latest job...")
    
    # 1. List all job folders
    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME, 
        Prefix='models/', 
        Delimiter='/'
    )
    
    if 'CommonPrefixes' not in response:
        print("❌ No job folders found.")
        return None
        
    # 2. Sort to find the newest one
    job_folders = [x['Prefix'] for x in response['CommonPrefixes']]
    latest_job = sorted(job_folders)[-1]
    
    print(f"📂 Found latest job: {latest_job}")
    
    # 3. Define the path (Standard SageMaker path)
    model_key = f"{latest_job}output/model.tar.gz"
    
    # 4. Download
    print(f"⬇️  Downloading: {model_key}")
    try:
        s3.download_file(BUCKET_NAME, model_key, 'model.tar.gz')
    except Exception as e:
        print(f"❌ Could not download. The path might be wrong or the file is missing.")
        print(f"   Error: {e}")
        return None
        
    # 5. Extract
    print("📦 Extracting...")
    try:
        with tarfile.open('model.tar.gz', 'r:gz') as tar:
            tar.extractall(path='./downloaded_model')
    except Exception as e:
        print(f"❌ Zip file is corrupted: {e}")
        return None
        
    return './downloaded_model/model.pth'

def predict(model_path):
    device = torch.device("cpu")
    model = Net().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print("\n✅ SUCCESS: Model loaded into PyTorch!")
    except Exception as e:
        print(f"❌ Model architecture mismatch: {e}")
        return

    # Test Prediction
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    prediction = output.argmax(dim=1, keepdim=True).item()
    
    print(f"🧠 Test Prediction: {prediction}")
    print(f"🚀 Status: SYSTEM FULLY OPERATIONAL")

if __name__ == "__main__":
    model_path = get_latest_job_model()
    if model_path and os.path.exists(model_path):
        predict(model_path)