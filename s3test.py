import boto3

s3_resource = boto3.resource('s3')

first_bucket_name = "mytestnami"
first_file_name = "snow.gif"
s3_resource.Object(first_bucket_name, first_file_name).download_file(
    f'/tmp/{first_file_name}')
