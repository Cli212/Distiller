#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
from six.moves import urllib
import zipfile
import gzip
 
def un_gz(file_name):
    
    # 获取文件的名称，去掉后缀名
    f_name = file_name.replace(".gz", "")
    # 开始解压
    g_file = gzip.GzipFile(file_name)
    #读取解压后的文件，并写入去掉后缀名的同名文件（即得到解压后的文件）
    open(f_name, "wb+").write(g_file.read())
    g_file.close()

TRAIN_DATA_URL = 'https://00f74ba44bd8099387b461fab47f758433a22618fe-apidata.googleusercontent.com/download/storage/v1/b/natural_questions/o/v1.0-simplified%2Fsimplified-nq-train.jsonl.gz?jk=AFshE3XAcKSuphXoX0o9I9sF2HhKjBoWXWqtgqGhGL6IQe10x8A4zde0z7mfqSIrcpA-cwHeXXtxNYle4HqbBWmxddoMyzeHdeoh65HZjOI0xvUPXgTh_ye0eV1vxZf9iEmjvOCDJ1SrOx_NvJgay3Urkn3fKpJT3vaevGsBuFUS0OmrosGqNVPAhrnK9GI5KHhFPvd9Hm9QUPr7NBYGShwle0Y_4lj303hvbu4CFO0Ab2abKs4iuqK0eVD-CdRMljBEKXWpT5khW8ktx17CCUThsJi75K9RwKiQiIc_jSjDPAluL_Z1pivTXGFLfdQahKoeJ-sQe7J7NRoINP3I6e-kkJs9Xkujc38hfSKYrqCbLFTSakn6NS3x-N7CXSnhPlLCZZCpSn6NJxR-WkiGjYLiWTZ7qEgXDbs4A1gCZBuE80g_nkMA0p-8NroEIaGXSIe3v7FGeNLKB6LNjeC2SKFnjs0Hr4UZA7EvQ7XhSWYebRsO935_0OZJz_iDKff73a4pY_-bxOP16xJg-eIgmU_ENiLQu8ckpmE2Sd4Eiv2MhC-Z22yj-ICSasb9XKaasxhWcdGgoLfhT54UCimi1s0jTW6eXOlkyMc43JujXeC0CP3bgq1PPZDbmGU6oYwxqm6y6dfT8o-ZjSBrRVMdlQ4LC90CeqoahELM1ZTUnOn4A1QxJ-V7uxKgcKNRBXeYex1b9j_CSOEVFC9fqcoI4xwG6s8Bf9ybpY3TPWVU4UMfQ4MB4-x8AiX4ppImywkoxycNZXFEYcYFf1VHDq7_ysjBsdlYvZAC8dXGhJ3HrXO5ipIh9In-QqHJvdAgJW0Z3d5dXAGhHMYr8AhnUG86rTslnQAVp4lNO5buklmHAoD-O35uqK-eHHTvd21yLwdx8J3qqr1lVLToaFvH2XULNPTM4r1fs_FzNNHzRdE&isca=1'
DEV_DATA_URL = 'https://00f74ba44b207748951b0848900738e6a71a3c8c50-apidata.googleusercontent.com/download/storage/v1/b/natural_questions/o/v1.0-simplified%2Fnq-dev-all.jsonl.gz?jk=AFshE3VOqTETbhMbDiPZ-rGl0fycQoFDRoVuEi64l-uDpmMfNcrCdjSxqcDfuthySWQh0w9I24CNYcBz3dBtaFUOZM5qz_olrpwEUARwLdnOe3LWpaswsOfhbh8GeOQq123LDcQjVqP1oe0o01hsCBj3OsBPIJlmOdIAVOXoET8I8XxBPHpUfdqJzkdvcXwG-n5i3nRhF-DQBaIqi-Pi6vhn1VuvbO4-233x8U93fbrgafqQCWG7vDSV60PxDtSnqW-OUNt_IGxfRLM2avZ3iGu0XWW3DE_-AWOJbTV5qjQCF74KuOA7i2ILPky3dVX4iTZnm-t0lb2ECl7sJsm0pXoXPEy4lfdXLmDsurULBeswPcSmOKt9P2cM8JyBN9TxifVef2TWnn2sMfkOmhwEVaorntlvzhECTjBa_r8Ae1kWOl5RCyFo2qCyHJyKY1yhhvZfDS0fB6mxehgiV6vL9ROVxyym-_yLAeIjHfunfEW_CvFWlK5hqn5_5sBEwE7MIY9zTIR185YoloVrol5GgCbwSvp6aZPj_orUROi5yCFSQnPWI0b-MRU5YsTWCcBlz1AOQgdjAq6uuTggabbW1UVLFnlPF0vlGcNVG70w8KjD9FbhFN_fN-kveDtfV-JAJCuM3Z_XOyvDC1DBMOW4fF2surJjxLu5IUvHsWjp1hMLPCgwNIHhgLhHSIvSSXPmWMQZNm-25gMguSkBNwc527ir71coY7QVKzG8bGFsEiHHNW10ELtdqjqr21snd-qT-69rBZ8QKkXzP0S6js2Yi6xXUM6N9jNZU3thhep7dONbeY9Fd5dSK2oyobOWRBneYcMnkyQS-N06mquSFqqYDVXZeWyGP-7fCrMU0ZNh3amjNK9aO90jQKepjTkrSMva0yAT8CXQXr6grkm8-ta_qqnSV2U&isca=1'
DATA_DIR = './datasets/NaturalQuestions'
TRAIN_DATA_NAME = 'simplified-nq-train.jsonl.gz'
DEV_DATA_NAME = 'nq-dev-all.jsonl.gz'
 
train = True
for url in [TRAIN_DATA_URL, DEV_DATA_URL]:
    filename = TRAIN_DATA_URL if train else DEV_DATA_URL
    filepath = os.path.join(DATA_DIR, filename)
    filepath, _ = urllib.request.urlretrieve(DEV_DATA_URL, filepath)
    un_gz(filepath)
    if train:
        train = False







