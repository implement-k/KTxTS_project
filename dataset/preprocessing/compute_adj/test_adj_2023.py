import pickle
import os

out_path = '/Users/implement/KT/KTDB/dataset/processed/dong_adjacency_2023.pkl'
with open(out_path, 'rb') as f:
    adj_dict = pickle.load(f)

manual_dong_mapping = {
    11230740: [11230810],              # 일원2동 -> 개포3동
    31101690: [31101740, 31101750],    # 행신3동 -> 행신3동, 행신4동
    31101700: [31101720, 31101730],    # 삼송동 -> 삼송1동, 삼송2동
    31103520: [31103620, 31103630],    # 중산동 -> 중산1동, 중산2동
    31104540: [31104600, 31104610],    # 탄현동 -> 탄현1동, 탄현2동
    31104590: [31104620, 31104630],    # 송산동 -> 덕이동, 가좌동
    31250110: [31250600, 31250610, 31250620, 31250630],  # 오포읍 -> 오포1동, 오포2동, 신현동, 능평동
}

# Apply manual mapping
for old_code, new_codes in manual_dong_mapping.items():
    if old_code in adj_dict:
        old_neighbors = adj_dict[old_code]
        del adj_dict[old_code]
        
        for new_code in new_codes:
            adj_dict[new_code] = list(old_neighbors) # Copy old neighbors
            
            # Add other new codes from the same split as neighbors
            for other_new_code in new_codes:
                if new_code != other_new_code:
                    adj_dict[new_code].append(other_new_code)
                    
        # Update all neighbors to point to the new codes instead of old_code
        for n in old_neighbors:
            if n in adj_dict:
                if old_code in adj_dict[n]:
                    adj_dict[n].remove(old_code)
                    adj_dict[n].extend(new_codes)

# Clean up duplicates in lists just in case
for k in adj_dict:
    adj_dict[k] = list(set(adj_dict[k]))
    
with open(out_path, 'wb') as f:
    pickle.dump(adj_dict, f)
    
print(f'Done! Saved {len(adj_dict)} nodes to {out_path}')

# Let's verify one of the splits
print("Neighbors of 31101720 (삼송1동):", adj_dict.get(31101720))
print("Neighbors of 31101730 (삼송2동):", adj_dict.get(31101730))
