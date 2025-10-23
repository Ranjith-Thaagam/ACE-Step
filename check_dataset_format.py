# check_dataset_format.py
from datasets import load_from_disk
import os

def check_dataset_format(dataset_path):
    """Check if the dataset matches Text2MusicDataset expectations"""
    print(f"🔍 Checking dataset format: {dataset_path}")
    
    try:
        dataset_dict = load_from_disk(dataset_path)
        
        print("✅ Dataset loaded successfully!")
        print(f"📊 Available splits: {list(dataset_dict.keys())}")
        
        # Check train split
        if 'train' in dataset_dict:
            train_dataset = dataset_dict['train']
            print(f"📊 Train samples: {len(train_dataset)}")
            
            # Check first sample structure
            if len(train_dataset) > 0:
                first_sample = train_dataset[0]
                print("🔍 First sample keys:", list(first_sample.keys()))
                
                # Check for required fields
                required_fields = ['audio', 'text']
                missing_fields = [field for field in required_fields if field not in first_sample]
                
                if missing_fields:
                    print(f"❌ Missing required fields: {missing_fields}")
                else:
                    print("✅ All required fields present!")
                    
                    # Check audio format
                    audio_data = first_sample['audio']
                    print("🔍 Audio keys:", list(audio_data.keys()) if isinstance(audio_data, dict) else "Not a dict")
                    
        else:
            print("❌ No 'train' split found!")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")

if __name__ == "__main__":
    check_dataset_format("./data/tamil_dataset_ace_format")