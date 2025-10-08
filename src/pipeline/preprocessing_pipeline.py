import json
import random
from sklearn.model_selection import train_test_split


with open("data/raw/medical_meadow_medqa.json", "r") as f:
    data = json.load(f)
    random.shuffle(data)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

with open("artifacts/train.json", 'w') as f:
    json.dump(train_data,f, indent=4)
    
    
with open("artifacts/validation.json", 'w') as f:
    json.dump(val_data,f, indent=4)
    
with open("artifacts/test.json", 'w') as f:
    json.dump(test_data,f, indent=4)
    
print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")
