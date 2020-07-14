import os
import sagemaker
import numpy as np
from sagemaker.tensorflow import TensorFlow


ON_SAGEMAKER_NOTEBOOK = False

sagemaker_session = sagemaker.Session()
if ON_SAGEMAKER_NOTEBOOK:
    role = sagemaker.get_execution_role()
else:
    role = "arn:aws:iam::761845552964:role/service-role/AmazonSageMaker-ExecutionRole-20200708T144653"


bucket = "forager-training-data"
train_instance_type='ml.p2.xlarge'      # The type of EC2 instance which will be used for training
deploy_instance_type='ml.p2.xlarge'     # The type of EC2 instance which will be used for deployment
hyperparameters={}

train_input_path = "s3://{}/Training/".format(bucket)
validation_input_path = "s3://{}/Validation/".format(bucket)

estimator = TensorFlow(
  entry_point=os.path.join(os.path.dirname(__file__), "train_mobilenet.py"),             
  role=role,
  framework_version="1.12.0",              
  hyperparameters=hyperparameters,
  training_steps=1000,
  evaluation_steps=100,
  train_instance_count=1,                   # "The number of GPUs instances to use"
  train_instance_type=train_instance_type,
)

print("Training ...")
estimator.fit({'training': train_input_path, 'eval': validation_input_path})