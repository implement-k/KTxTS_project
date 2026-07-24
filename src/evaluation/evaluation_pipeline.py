import torch
import numpy as np
from sklearn.metrics import mean_squared_error

def cpc_score(y_true, y_pred):
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0:
        return 0.0
    return numerator / denominator

def run_evaluation_pipeline(model, data_dict, device, model_type='mae', criterion=None):
    """
    data_dict: {CityName: {TaskID: [sample_dict, ...]}}
    Returns:
        results: {CityName: {TaskID: {'rmse': float, 'cpc': float, 'loss': float}}}
    """
    if model is not None:
        model.eval()
    is_hybrid_od = False
    if criterion is not None and criterion.__class__.__name__ == 'HybridWeightedODLoss':
        is_hybrid_od = True
        
    results = {}
    
    with torch.no_grad():
        for city, tasks in data_dict.items():
            results[city] = {}
            for task_id, samples in tasks.items():
                task_rmse = []
                task_cpc = []
                task_loss = []
                
                for batch in samples:
                    x_static = batch['X_static'].unsqueeze(0).to(device)
                    x_d = batch['X_dist'].unsqueeze(0).to(device)
                    x_o = batch['X_OD_masked'].unsqueeze(0).to(device)
                    y_o = batch['y_OD'].unsqueeze(0).to(device)
                    input_mask = batch['mask'].unsqueeze(0).to(device)
                    active_node_mask = batch['active_node_mask'].unsqueeze(0).to(device)
                    loss_mask = batch['loss_mask'].unsqueeze(0).to(device)
                    
                    if model_type == 'mae':
                        if is_hybrid_od:
                            pred_scale, pred_raw = model(x_static, x_o, x_d, input_mask, active_node_mask)
                            m2d = loss_mask.unsqueeze(1) | loss_mask.unsqueeze(2)
                            active_m2d = active_node_mask.unsqueeze(1) & active_node_mask.unsqueeze(2)
                            valid_cells = (m2d & active_m2d).cpu().numpy()[0]
                            
                            diag_mask = torch.eye(pred_scale.size(1), dtype=torch.bool, device=device).unsqueeze(0)
                            if criterion is not None:
                                v_loss = criterion(pred_scale, pred_raw, y_o, 1.0, m2d, diag_mask, active_node_mask=active_node_mask).item()
                            else:
                                v_loss = 0.0
                                
                            p_real = np.maximum(torch.expm1(pred_scale[0].cpu()).numpy()[valid_cells], 0)
                            
                        else:
                            pred_raw = model(x_static, x_o, x_d, input_mask, active_node_mask)
                            m2d = loss_mask.unsqueeze(1) | loss_mask.unsqueeze(2)
                            active_m2d = active_node_mask.unsqueeze(1) & active_node_mask.unsqueeze(2)
                            valid_cells = (m2d & active_m2d).cpu().numpy()[0]
                            
                            if criterion is not None:
                                v_loss = criterion(pred_raw, y_o, 1.0, mask=m2d).item()
                            else:
                                v_loss = 0.0
                                
                            p_real = np.maximum(torch.expm1(pred_raw[0].cpu()).numpy()[valid_cells], 0)
                            
                    elif model_type == 'gravity':
                        # Dummy for gravity
                        p_real = np.zeros(1)
                        v_loss = 0.0
                        valid_cells = [0]
                    
                    y_real = np.maximum(torch.expm1(y_o[0].cpu()).numpy()[valid_cells], 0)
                    
                    p_real = np.nan_to_num(p_real, nan=0.0, posinf=1e10, neginf=0.0)
                    
                    if len(y_real) > 0:
                        rmse = np.sqrt(mean_squared_error(y_real, p_real))
                        cpc = cpc_score(y_real, p_real)
                    else:
                        rmse, cpc = 0.0, 0.0
                        
                    task_rmse.append(rmse)
                    task_cpc.append(cpc)
                    task_loss.append(v_loss)
                    
                results[city][task_id] = {
                    'rmse': np.mean(task_rmse),
                    'cpc': np.mean(task_cpc),
                    'loss': np.mean(task_loss)
                }
                
    return results
