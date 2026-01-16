import boto3
import os
from torchvision import datasets
from torchvision import transforms
from dotenv import load_dotenv 


load_dotenv() 

BUCKET_NAME = os.getenv('BUCKET_NAME') 
S3_PREFIX = 'data/raw'
LOCAL_DIR = './data'

def upload_to_s3(local_path, s3_path):

    s3 = boto3.client('s3')
    
    try:
        s3.upload_file(local_path, BUCKET_NAME, s3_path)
        print(f"✅ Uploaded {local_path} to s3://{BUCKET_NAME}/{s3_path}")
    except Exception as e:
        print(f"❌ Failed to upload {local_path}: {e}")

def main():
    if not BUCKET_NAME:
        print("❌ Error: BUCKET_NAME not found in .env file")
        return

    print(f"Target Bucket: {BUCKET_NAME}")
    print("⬇️  Downloading MNIST data locally...")
    
    train_data = datasets.MNIST(root=LOCAL_DIR, train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root=LOCAL_DIR, train=False, download=True, transform=transforms.ToTensor())

    print("⬆️  Uploading to S3...")
    
    base_path = os.path.join(LOCAL_DIR, 'MNIST', 'raw')
    files_to_upload = [
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte'
    ]

    for file_name in files_to_upload:
        local_file = os.path.join(base_path, file_name)
        s3_dest = f"{S3_PREFIX}/{file_name}"
        upload_to_s3(local_file, s3_dest)

    print(f"\n🚀 Success! Data is now in s3://{BUCKET_NAME}/{S3_PREFIX}")

if __name__ == "__main__":
    main()