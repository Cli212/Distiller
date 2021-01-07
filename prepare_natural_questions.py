import json
import boto3
from io import BytesIO
import gzip
import os
import argparse
import ast

ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
TRAIN_DATA_URL = 'NaturalQuestions/v1.0-simplified_simplified-nq-train.jsonl.gz'
DEV_DATA_URL = 'NaturalQuestions/nq-dev-all.jsonl.gz'
DATA_DIR = './datasets/NaturalQuestions'
TRAIN_DATA_NAME = 'simplified-nq-train.jsonl.gz'
DEV_DATA_NAME = 'nq-dev-all.jsonl.gz'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all',default=False,type=ast.literal_eval)
    args = parser.parse_args()
    if args.all:
        pass
    else:
        first = True
        for url in [DEV_DATA_URL, TRAIN_DATA_URL]:
            filename = TRAIN_DATA_NAME if first else DEV_DATA_NAME
            try:
                 s3 = boto3.resource('s3')
                 key=url
                 obj = s3.Object('gluonnlp-numpy-data',key)
                 n = obj.get()['Body'].read()
                 gzipfile = BytesIO(n)
                 gzipfile = gzip.GzipFile(fileobj=gzipfile)
                 content = gzipfile.read()
                 open(os.path.join(DATA_DIR, filename), "wb+").write(content)
            except Exception as e:
                print(e)
                raise e




