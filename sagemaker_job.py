import os
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

bucket_name = os.getenv('BUCKET_NAME')
role_arn = os.getenv('SAGEMAKER_ROLE_ARN')
region = os.getenv('AWS_REGION')

# --- NEW SECTION: EXPLICIT SESSION SETUP ---
# This forces SageMaker to see your .env region, fixing the ValueError
boto_session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=region
)
sagemaker_session = sagemaker.Session(boto_session=boto_session)
# -------------------------------------------

print(f"🚀 Preparing to launch training job in {region}...")
print(f"   Bucket: {bucket_name}")
print(f"   Role:   {role_arn}")

# 2. Define the Estimator
estimator = PyTorch(
    entry_point='train.py',
    source_dir='src',
    role=role_arn,
    framework_version='2.0',
    py_version='py310',
    instance_count=1,
    instance_type='ml.m5.large',
    
    # CRITICAL: Pass the custom session here!
    sagemaker_session=sagemaker_session, 
    
    hyperparameters={
        'epochs': 1,
        'batch-size': 64,
        'learning-rate': 0.01
    },
    output_path=f"s3://{bucket_name}/models/"
)

# 3. Define Data Inputs
data_channels = {
    'training': f"s3://{bucket_name}/data/raw"
}

# 4. Launch the Job
print("\n☁️  Submitting job to AWS SageMaker... (This will take a few minutes)")
estimator.fit(data_channels)

print(f"✅ Job finished! Model saved to s3://{bucket_name}/models/")