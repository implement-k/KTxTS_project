def build_mapping(present_dongs, rules):
    # rules: dict of A -> [list of targets]
    # To handle cascaded rules properly, we process top-down.
    
    mapping = {d: [d] for d in present_dongs}
    
    # We iteratively apply rules until no more changes.
    changed = True
    while changed:
        changed = False
        for d in present_dongs:
            current_targets = mapping[d]
            new_targets = []
            for t in current_targets:
                if t in rules:
                    # check if any of the rule's newly spawned targets are ALREADY in present_dongs
                    # wait, if A -> A, B. B is the newly spawned.
                    # What if A -> C, D? Then C and D are newly spawned.
                    # General rule: if a node t splits into targets, and ANY of the strictly NEW targets
                    # (targets other than t itself) is present in the dataset natively,
                    # it means the split is already reflected in the data. So we don't apply the split.
                    # Wait, what if ALL targets are new (e.g. A -> B, C)?
                    # If A -> B, C, and B is present, but C is absent. The data has B. Where is C?
                    # This is complex. Let's assume A -> A, B format for now.
                    
                    split_targets = rules[t]
                    newly_spawned = set(split_targets) - {t}
                    
                    # If NONE of the newly spawned targets are in present_dongs, we apply the rule.
                    if not any(n in present_dongs for n in newly_spawned):
                        if set(split_targets) != set([t]): # prevent infinite loop if no change
                            new_targets.extend(split_targets)
                            changed = True
                            continue
                new_targets.append(t)
            
            # remove duplicates but preserve order
            seen = set()
            dedup = []
            for nt in new_targets:
                if nt not in seen:
                    seen.add(nt)
                    dedup.append(nt)
            mapping[d] = dedup
            
    return mapping

rules = {
    '송도2동': ['송도2동', '송도4동'],
    '송도4동': ['송도4동', '송도5동']
}

print("Case 1: Only 송도2동 is present")
print(build_mapping({'송도2동'}, rules))

print("\nCase 2: 송도2동 and 송도4동 are present")
print(build_mapping({'송도2동', '송도4동'}, rules))

print("\nCase 3: All three are present")
print(build_mapping({'송도2동', '송도4동', '송도5동'}, rules))
