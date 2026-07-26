import os
import random
import json

SOURCE_DIR = 'data/card_images'
OUTPUT_JSON = 'data/split.json'
SEED = 42

SPLIT_RATIOS = {
    'train': 0.70,
    'val': 0.15,
    'test': 0.15
}

def create_split_json():
    
    # Reproducibility
    random.seed(SEED)

    all_files = sorted([f for f in os.listdir(SOURCE_DIR)])

    total_files = len(all_files)
    random.shuffle(all_files)
    # Indizes for Slicing
    train_end = int(total_files * SPLIT_RATIOS['train'])
    val_end = train_end + int(total_files * SPLIT_RATIOS['val'])

    # Slicing
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    # Build Dictionary
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    # Write in JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=4, ensure_ascii=False)

    print("\n--- Split erfolgreich erstellt ---")
    print(f" Train: {len(train_files)} Karten ({len(train_files)/total_files:.1%})")
    print(f" Val:   {len(val_files)} Karten ({len(val_files)/total_files:.1%})")
    print(f" Test:  {len(test_files)} Karten ({len(test_files)/total_files:.1%})")
    print(f"Gespeichert unter: {OUTPUT_JSON}\n")

if __name__ == '__main__':
    create_split_json()