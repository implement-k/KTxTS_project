import sys
import os
import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'mae-old'))
from dataset import ODDataset

import importlib.util
spec = importlib.util.spec_from_file_location("gravity_model", os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'gravity(경훈)', 'model.py'))
gravity_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gravity_module)
DoublyConstrainedGravityModel = gravity_module.DoublyConstrainedGravityModel

def get_trained_gravity_model(weight_path='dataset/gravity_model.pkl'):
    if os.path.exists(weight_path):
        print("Loading pre-trained gravity model...")
        return joblib.load(weight_path)
    
    print("Pre-trained gravity model not found. Training on the fly...")
    train_dataset = ODDataset(mode='train')
    
    y_train_raw = train_dataset.X_OD_raw
    X_static_train = train_dataset.X_static
    
    N = train_dataset.num_nodes
    self_train = np.diag(y_train_raw).copy()
    
    y_train_ext = y_train_raw.copy()
    np.fill_diagonal(y_train_ext, 0)
    O_train = y_train_ext.sum(axis=1)
    D_train = y_train_ext.sum(axis=0)
    inter_train = y_train_ext.sum(axis=1) # Or whatever it was
    
    model = DoublyConstrainedGravityModel(beta=2.0, max_iter=100)
    model.fit_lgbm_O_D(X_static_train, O_train, D_train)
    model.fit_lgbm_self_inter(X_static_train, self_train, inter_train)
    
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    joblib.dump(model, weight_path)
    print(f"Saved trained gravity model to {weight_path}")
    
    return model
