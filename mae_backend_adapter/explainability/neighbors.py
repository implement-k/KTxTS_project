from __future__ import annotations

import torch


def get_neighbors(a_spatial: torch.Tensor, city_codes: list[str], target_codes: list[str]) -> dict[str, list[str]]:
    if a_spatial.ndim != 2:
        raise ValueError(f"a_spatial은 (N, N)이어야 합니다. 현재 shape={tuple(a_spatial.shape)}")
    if a_spatial.shape[0] != a_spatial.shape[1]:
        raise ValueError("a_spatial은 정방행렬이어야 합니다.")
    if len(city_codes) != a_spatial.shape[0]:
        raise ValueError("city_codes 수와 adjacency node 수가 다릅니다.")
    city_codes = [str(code) for code in city_codes]
    code_to_idx = {code: idx for idx, code in enumerate(city_codes)}
    result = {}
    for target_code in target_codes:
        target_code = str(target_code)
        if target_code not in code_to_idx:
            result[target_code] = []
            continue
        target_idx = code_to_idx[target_code]
        neighbor_indices = torch.where(a_spatial[target_idx] > 0)[0].tolist()
        result[target_code] = [city_codes[idx] for idx in neighbor_indices if idx != target_idx]
    return result
