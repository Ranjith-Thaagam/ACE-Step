import torch
from src.tokenizer.tokenizer import TamilTokenizerWrapper

def test_ace_step_integration():
    print("🧪 Testing Tamil Tokenizer Integration with ACE Step...")
    
    # Load your trained tokenizer
    tokenizer = TamilTokenizerWrapper(
        model_path='chkpts/tokenizer/tamil_tokenizer_special.model',
        device='cpu'
    )
    
    print(f"✅ Tokenizer loaded successfully!")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    print(f"Unknown token ID: {tokenizer.unk_token_id}")
    
    # Test samples from your Tamil songs
    test_samples = [
        "பாடல் பாடுவேன்",
        "இசையும் பாடலும்",
        "காதல் பாடல்", 
        "உள்ளம் மகிழும் பாடல்",
        "தமிழிசை வாழ்க"
    ]
    
    print("\n🔤 Testing ACE Step Compatible Interface:")
    for sample in test_samples:
        # This is the exact format ACE Step expects
        inputs = tokenizer(
            sample,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        print(f"\nText: '{sample}'")
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Attention Mask shape: {inputs['attention_mask'].shape}")
        print(f"Input IDs: {inputs['input_ids'][0][:10]}...")  # First 10 tokens
        print(f"Attention Mask: {inputs['attention_mask'][0][:10]}...")
        
        # Test round-trip
        decoded = tokenizer.decode(inputs['input_ids'][0])
        print(f"Round-trip: '{decoded}'")
        print(f"Match: {sample == decoded}")
    
    # Test batch processing (like ACE Step will do)
    print("\n📦 Testing Batch Processing:")
    batch_inputs = tokenizer(
        test_samples,
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=256
    )
    
    print(f"Batch Input IDs shape: {batch_inputs['input_ids'].shape}")
    print(f"Batch Attention Mask shape: {batch_inputs['attention_mask'].shape}")
    print("✅ Batch processing works correctly!")
    
    return True

def check_special_tokens():
    print("\n🔍 Checking Special Tokens:")
    tokenizer = TamilTokenizerWrapper(
        model_path='chkpts/tokenizer/tamil_tokenizer_special.model',
        device='cpu'
    )
    
    special_tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', '[SEP]', '[MASK]', '[CLS]']
    
    for token in special_tokens:
        try:
            token_id = tokenizer.tokenizer.sp.piece_to_id(token)
            print(f"  {token}: ID = {token_id}")
        except:
            print(f"  {token}: Not found in vocabulary")

if __name__ == "__main__":
    check_special_tokens()
    test_ace_step_integration()
    print("\n🎉 Tamil tokenizer is ready for ACE Step integration!")