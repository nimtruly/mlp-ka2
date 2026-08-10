"""
analyze_test_leak.py — Analyze if customer_ids in test represent the same customers as in train.
"""
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

overlap_ids = set(train['customer_id']).intersection(set(test['customer_id']))
print('Number of overlapping customer_ids:', len(overlap_ids))

count_match_all = 0
count_total = 0

for cid in list(overlap_ids)[:100]:
    tr_rec = train[train['customer_id'] == cid]
    te_rec = test[test['customer_id'] == cid]
    
    for _, te_row in te_rec.iterrows():
        count_total += 1
        # Check if there is any row in tr_rec that has the same last_name, country, gender
        match = tr_rec[
            (tr_rec['last_name'] == te_row['last_name']) &
            (tr_rec['country'] == te_row['country']) &
            (tr_rec['gender'] == te_row['gender'])
        ]
        if len(match) > 0:
            count_match_all += 1

print(f'Overlapping records where name, country, gender match: {count_match_all} out of {count_total} ({count_match_all/count_total*100:.2f}%)')
