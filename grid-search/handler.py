import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import json
import paramiko 

def handle(req):
    data = json.loads(req)
    subspaces = data.get("subspace")
    num_samples = 100
    sampled_subspaces = random.sample(subspaces, num_samples)
    best_rmse = float('inf')
    best_r_squared = 0
    best_params = None
    master_ip = "172.22.85.50"
    master_username = "jet-2"
    master_password = "jet1234"  

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(master_ip, username=master_username, password=master_password)
    sftp = ssh.open_sftp()
    sftp.get('/home/jet-2/scaled_features.csv', 'scaled_features.csv')
    sftp.get('/home/jet-2/y.csv', 'y.csv')
    sftp.close()
    X = pd.read_csv('scaled_features.csv')
    y = pd.read_csv('y.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    for subspace in sampled_subspaces:
        rf = RandomForestRegressor()

        subspace['max_depth'] = None if subspace['max_depth'] == 'None' else int(subspace['max_depth'])
        rf.set_params(**subspace)

        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r_squared = rf.score(X_test, y_test)

        if rmse < best_rmse:
            best_rmse = rmse
            best_r_squared = r_squared
            best_params = subspace

    return {"best_params":best_params, "best_rmse":best_rmse, "best_r_squared":best_r_squared}




