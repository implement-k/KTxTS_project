import torch
import numpy as np
from sklearn.metrics import mean_squared_error

def cpc_score(y_true, y_pred):
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0:
        return 0.0
    return numerator / denominator

def run_evaluation_pipeline(model, data_dict, device, model_type='mae', criterion=None, **kwargs):
    """
    data_dict: {CityName: {TaskID: [sample_dict, ...]}}
    Returns:
        results: {CityName: {TaskID: {'rmse': float, 'cpc': float, 'prmse': float, 'loss': float}}}
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
                task_prmse = []
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
                        
                        import sys
                        import os
                        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gravity(경훈)'))
                        from model import DoublyConstrainedGravityModel
                        
                        g_type = kwargs.get('gravity_model_type', 'lgbm')
                        g_imp = kwargs.get('gravity_imputation', 'zero')
                        
                        grav_model = DoublyConstrainedGravityModel(generation_model_type=g_type, beta=2.0, max_iter=100)
                        
                        # Data prep
                        mask_np = batch['mask'][0].cpu().numpy()
                        train_mask_grav = ~mask_np
                        y_OD_raw = batch['y_OD_raw'][0].cpu().numpy()
                        X_dist_grav = batch['X_dist'][0].cpu().numpy()
                        
                        X_o = y_OD_raw[train_mask_grav].sum(axis=1) - np.diag(y_OD_raw)[train_mask_grav]
                        X_d = y_OD_raw[:, train_mask_grav].sum(axis=0) - np.diag(y_OD_raw)[train_mask_grav]
                        X_self = np.diag(y_OD_raw)[train_mask_grav]
                        X_inter = X_o # For simplified self-loop fallback
                        
                        # Handle imputation on masking indices
                        X_static_masked = batch['X_static'][0].cpu().numpy().copy()
                        X_static_raw_masked = batch['X_static_raw'][0].cpu().numpy().copy()
                        
                        if g_imp == 'mean':
                            import sys
                            import os
                            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mae-old'))
                            from dataset import ODDataset
                            dummy_ds = ODDataset(mode='val')
                            
                            train_means = X_static_masked[train_mask_grav].mean(axis=0)
                            train_means_raw = X_static_raw_masked[train_mask_grav].mean(axis=0)
                            for c in dummy_ds.masking_indices:
                                X_static_masked[mask_np, c] = train_means[c]
                                X_static_raw_masked[mask_np, c] = train_means_raw[c]
                        
                        X_static_all = X_static_raw_masked if g_type in ['trip_rate', 'cross_class', 'linear_regression'] else X_static_masked
                        
                        # Remove indicator columns (is_masked, is_merged) before passing to model
                        X_static_train_grav = X_static_masked[train_mask_grav, :-2]
                        X_static_all_grav = X_static_all[:, :-2]
                        
                        p_grav = grav_model.fit_predict(
                            X_static_train=X_static_train_grav,
                            O_train=X_o,
                            D_train=X_d,
                            X_static_all=X_static_all_grav,
                            dist_matrix=X_dist_grav
                        )
                        
                        p_real = np.maximum(p_grav[valid_cells], 0)
                        v_loss = 0.0
                    
                    y_real = np.maximum(torch.expm1(y_o[0].cpu()).numpy()[valid_cells], 0)
                    
                    p_real = np.nan_to_num(p_real, nan=0.0, posinf=1e10, neginf=0.0)
                    
                    if len(y_real) > 0:
                        rmse = np.sqrt(mean_squared_error(y_real, p_real))
                        cpc = cpc_score(y_real, p_real)
                        mean_y = np.mean(y_real)
                        prmse = rmse / mean_y if mean_y > 0 else 0.0
                    else:
                        rmse, cpc, prmse = 0.0, 0.0, 0.0
                        
                    task_rmse.append(rmse)
                    task_cpc.append(cpc)
                    task_prmse.append(prmse)
                    task_loss.append(v_loss)
                    
                results[city][task_id] = {
                    'rmse': np.mean(task_rmse),
                    'cpc': np.mean(task_cpc),
                    'prmse': np.mean(task_prmse),
                    'loss': np.mean(task_loss)
                }
                
    return results
