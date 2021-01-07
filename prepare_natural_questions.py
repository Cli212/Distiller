import json
import boto3
from io import BytesIO
import gzip
import os

ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
TRAIN_DATA_URL = 'https://gluonnlp-numpy-data.s3-us-west-2.amazonaws.com/NaturalQuestions/v1.0-simplified_simplified-nq-train.jsonl.gz'
DEV_DATA_URL = 'https://gluonnlp-numpy-data.s3-us-west-2.amazonaws.com/NaturalQuestions/nq-dev-all.jsonl.gz'
DATA_DIR = './datasets/NaturalQuestions'
TRAIN_DATA_NAME = 'simplified-nq-train.jsonl.gz'
DEV_DATA_NAME = 'nq-dev-all.jsonl.gz'

first = True
for url in [TRAIN_DATA_URL, DEV_DATA_URL]:
    filename = TRAIN_DATA_NAME if first else DEV_DATA_NAME
    try:
         s3 = boto3.resource('s3')
         key='v1.0-simplified_simplified-nq-train.jsonl.gz'
         obj = s3.Object('gluonnlp-numpy-data',key)
         n = obj.get()['Body'].read()
         gzipfile = BytesIO(n)
         gzipfile = gzip.GzipFile(fileobj=gzipfile)
         content = gzipfile.read()
         open(os.path.join(DATA_DIR, filename), "wb+").write(content)
    except Exception as e:
        print(e)
        raise e

# import gzip
 
# def un_gz(file_name):
#
#     # 获取文件的名称，去掉后缀名
#     f_name = file_name.replace(".gz", "")
#     # 开始解压
#     g_file = gzip.GzipFile(file_name)
#     #读取解压后的文件，并写入去掉后缀名的同名文件（即得到解压后的文件）
#     open(f_name, "wb+").write(g_file.read())
#     g_file.close()


 
# train = True
# for url in [TRAIN_DATA_URL, DEV_DATA_URL]:
#     filename = TRAIN_DATA_URL if train else DEV_DATA_URL
#     filepath = os.path.join(DATA_DIR, filename)
#     filepath, _ = urllib.request.urlretrieve(DEV_DATA_URL, filepath)
#     un_gz(filepath)
#     if train:
#         train = False







