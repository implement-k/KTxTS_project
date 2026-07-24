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
    if criterion is not None and criterion.__class__.__name__ == 'HybridWeightedMSELoss':
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
                    
                    if model_type in ['mae', 'mae-old']:
                        if model is None:
                            continue
                            
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
                    
                    elif model_type == 'mae-new':
                        if model is None:
                            continue
                            
                        idx_cont = model.split_indices['cont']
                        idx_prop_multi = model.split_indices['prop_multi']
                        idx_prop_single = model.split_indices['prop_single']
                        idx_zero = model.split_indices['zero']
                        
                        x_cont = x_static[:, :, idx_cont]
                        x_prop_multi = x_static[:, :, idx_prop_multi]
                        x_prop_single = x_static[:, :, idx_prop_single]
                        x_zero = x_static[:, :, idx_zero]
                        
                        if is_hybrid_od:
                            pred_scale, pred_raw = model(x_cont, x_prop_multi, x_prop_single, x_zero, x_o, x_d, input_mask, active_node_mask)
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
                            pred_raw = model(x_cont, x_prop_multi, x_prop_single, x_zero, x_o, x_d, input_mask, active_node_mask)
                            m2d = loss_mask.unsqueeze(1) | loss_mask.unsqueeze(2)
                            active_m2d = active_node_mask.unsqueeze(1) & active_node_mask.unsqueeze(2)
                            valid_cells = (m2d & active_m2d).cpu().numpy()[0]
                            
                            if criterion is not None:
                                v_loss = criterion(pred_raw, y_o, 1.0, mask=m2d).item()
                            else:
                                v_loss = 0.0
                                
                            p_real = np.maximum(torch.expm1(pred_raw[0].cpu()).numpy()[valid_cells], 0)
                            
                    elif model_type == 'twostage':
                        if model is None:
                            continue
                        
                        m2d = loss_mask.unsqueeze(1) | loss_mask.unsqueeze(2)
                        active_m2d = active_node_mask.unsqueeze(1) & active_node_mask.unsqueeze(2)
                        valid_cells = (m2d & active_m2d).cpu().numpy()[0]
                        
                        # Generate Stage 1 predictions
                        if hasattr(model, 'stage1_model'):
                            N = x_static.shape[1]
                            X_static_np = x_static.cpu().numpy()[0]
                            mask_idx = np.where(input_mask.cpu().numpy()[0])[0]
                            # Using fit_predict logic, but just predicting here
                            stage1_o_pred, stage1_d_pred = model.stage1_model.predict(X_static_np, mask_idx)
                            
                            stage1_o = torch.tensor(stage1_o_pred, dtype=torch.float32, device=device).unsqueeze(0)
                            stage1_d = torch.tensor(stage1_d_pred, dtype=torch.float32, device=device).unsqueeze(0)
                        else:
                            # Fallback if stage 1 model is missing
                            stage1_o = torch.zeros((1, N), device=device)
                            stage1_d = torch.zeros((1, N), device=device)
                            
                        pred_raw = model(x_static, stage1_o, stage1_d, x_d)
                        
                        if criterion is not None:
                            v_loss = criterion(pred_raw, y_o, 1.0, mask=m2d).item()
                        else:
                            v_loss = 0.0
                            
                        p_real = np.maximum(torch.expm1(pred_raw[0].cpu()).numpy()[valid_cells], 0)

                    elif model_type == 'gravity':
                        m2d = loss_mask.unsqueeze(1) | loss_mask.unsqueeze(2)
                        active_m2d = active_node_mask.unsqueeze(1) & active_node_mask.unsqueeze(2)
                        valid_cells = (m2d & active_m2d).cpu().numpy()[0]
                        
                        if 'p_gravity' in batch:
                            p_grav = batch['p_gravity']
                            p_real = np.maximum(p_grav[valid_cells], 0)
                        else:
                            p_real = np.zeros(np.sum(valid_cells))
                            
                        v_loss = 0.0
                    
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
