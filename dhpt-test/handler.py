import redis
import time
import numpy as np
import pickle
import pandas as pd
import paramiko
import json
def handle(res):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("172.22.85.50", username="jet-2", password="jet1234")
    sftp = ssh.open_sftp()
    sftp.get("matrix1.csv", "matrix1.csv")
    sftp.close()

    redis_client = redis.Redis(host='172.22.85.64', port=6379)
    timeout = 600
    start_time = time.time()
    checkpoint_key = "matrix_multiplication_checkpoint7"
    checkpoint_data = redis_client.get(checkpoint_key)

    if checkpoint_data is not None:
        checkpoint_data = pickle.loads(checkpoint_data)
        A1 = checkpoint_data["A1"]
        A2 = checkpoint_data["A2"]
        A3 = checkpoint_data["A3"]
        A4 = checkpoint_data["A4"]
        C1 = checkpoint_data["C1"]
        start_time = checkpoint_data["start_time"]
    else:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect("172.22.85.50", username="jet-2", password="jet1234")
        sftp = ssh.open_sftp()
        sftp.get("matrix1.csv", "matrix1.csv")
        sftp.close()
        A1 = pd.read_csv("matrix1.csv",header=None)
        A2 = pd.read_csv("matrix1.csv",header=None)
        A3 = pd.read_csv("matrix1.csv",header=None)
        A4 = pd.read_csv("matrix1.csv",header=None)
        C1 = A1.dot(A2)

    if (time.time() - start_time) >= (timeout - 30):
        checkpoint_data = {
            "A1": A1,
            "A2": A2,
            "A3": A3,
            "A4": A4,
            "C1": C1,
            "start_time": start_time
        }

        checkpoint_data_serialized = pickle.dumps(checkpoint_data)
        redis_client.set(checkpoint_key, checkpoint_data_serialized)

    C2 = A3.dot(A4)
    a1a2_plus_a3a4 = C1 + C2
    sum_elements = str(a1a2_plus_a3a4.sum().sum())
    return json.dumps({"statusCode": 200, "body": sum_elements})




