# Synthetic Data Label Changes

## Summary
On October 26, 2026, we updated 301 synthetic feedback rows from HIGH to MEDIUM severity to align with the refined safety escalation policy.

## Policy Decision (Task 4)
The updated policy distinguishes between:
- **HIGH severity**: Food contamination with pests, washroom/toilet hygiene issues, allergen label errors, staff misconduct, unsafe products
- **MEDIUM severity**: General floor/shelf cleanliness, trash visible, messy aisles

## What Changed
**Category affected**: Store Cleanliness  
**Number of rows**: 301  
**Old severity**: HIGH  
**New severity**: MEDIUM  
**Date**: 2026-10-26

## Explanation of the 301 vs 239 discrepancy
The initial estimate of "about 239 rows" referred to rows matching specific patterns like "dirty floor" and "trash visible". The actual count of 301 includes the full Store Cleanliness category, which contains additional keywords:
- "dirty floor", "trash visible" (the ~239 estimated)
- "messy", "cleaning missed", "entrance trash" (additional patterns)
- Transliterated variations in Hindi, Tamil, Telugu, Marathi

All 301 Store Cleanliness rows are now MEDIUM, consistent with the policy that general floor/trash cleanliness is operational (MEDIUM), not safety-critical (HIGH).

## Sample of Changed Rows (10 of 301)
```
1. "Entrance daggara trash undi. asap" → MEDIUM
2. "Cleaning was missed in the aisles. urgent" → MEDIUM
3. "Entrance jawal kachra aahe. urgent" → MEDIUM (Marathi)
4. "Entrance jawal kachra aahe. 😟" → MEDIUM
5. "The store floor is dirty and trash is visible near the entrance. asap" → MEDIUM
6. "Store floor dirty ah irukku. !!" → MEDIUM (Tamil)
7. "Cleaning was missed in the aisles. 😟" → MEDIUM
8. "The store floor is dirty and trash is visible near the entrance. 😟" → MEDIUM
9. "Store floor dirty ga undi. !!" → MEDIUM (Telugu)
10. "Entrance daggara trash undi." → MEDIUM
```

## Impact on Metrics

### Before Label Update (Policy forcing HIGH on Store Cleanliness):
- Escalation precision: ~0.5 (many false HIGH escalations)
- Policy conflicted with synthetic labels (labels said MEDIUM, policy forced HIGH)

### After Label Update (Labels aligned with policy):
- Escalation precision: Improved (labels now match policy intent)
- Store Cleanliness correctly categorized as MEDIUM operational issue
- Washroom/toilet hygiene (with negative qualifiers) remains HIGH

## Important Note
**This improved escalation precision reflects a label correction, not a model improvement.** The synthetic labels were updated to match the intended policy. The model's behavior improved because:
1. Labels now correctly reflect the policy (floor cleanliness = MEDIUM)
2. Policy engine correctly implements the distinction (washroom hygiene = HIGH, floor cleanliness = MEDIUM)

## Verification
The policy implementation was tested with:
- "Store floor is dirty and trash is visible" → MEDIUM (Store Cleanliness)
- "Washroom is dirty and smells bad" → HIGH (Washroom Hygiene)
- "Garbage is overflowing and smelly" → HIGH (Washroom Hygiene)

All tests pass, confirming policy is correctly implemented.

## Data Integrity
The change was made programmatically:
```python
df.loc[df['issue_category']=='Store Cleanliness', 'severity'] = 'MEDIUM'
```

Original data structure preserved; only severity column updated for Store Cleanliness rows.
