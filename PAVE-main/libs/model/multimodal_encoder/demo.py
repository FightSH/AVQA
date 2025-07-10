import torch
from transformers import AutoModel, AutoImageProcessor, logging, CLIPVisionModel

# Suppress verbose logging from transformers to keep the output clean
logging.set_verbosity_error()

def check_model_for_token_merge(model_name_or_path: str) -> bool:
    """
    Checks if a Hugging Face model is a Vision Transformer (ViT) architecture
    that includes a [CLS]-like token, making it suitable for the token merge strategy.

    This robust version correctly handles:
    1. Standard ViT Models (e.g., 'google/vit-base-patch16-224-in21k').
    2. CLIP Vision Models (e.g., 'openai/clip-vit-large-patch14'), by explicitly
       loading only the vision encoder to ensure a consistent output structure.

    Args:
        model_name_or_path: The identifier of the model on the Hugging Face Hub
                            or a path to a local model directory.

    Returns:
        True if the model is compatible, False otherwise.
    """
    print(f"\n--- Checking model: {model_name_or_path} ---")
    try:
        # --- Step 1: Load Model and Processor Intelligently ---
        is_clip_model = 'clip' in model_name_or_path.lower()
        if is_clip_model:
            print("ℹ️ INFO: Detected CLIP model. Loading vision encoder specifically.")
            model = CLIPVisionModel.from_pretrained(model_name_or_path,cache_dir='/mnt/sda/shenhao/models')
        else:
            model = CLIPVisionModel.from_pretrained(model_name_or_path, cache_dir='/mnt/sda/shenhao/models')
            # model = AutoModel.from_pretrained(model_name_or_path,cache_dir='/mnt/sda/shenhao/models')
        
        config = model.config
        processor = AutoImageProcessor.from_pretrained(model_name_or_path)

        # --- Step 2: Basic ViT Architecture Checks ---
        if not hasattr(config, 'patch_size') or not (hasattr(processor, 'size') or hasattr(processor, 'crop_size')):
            print("❌ FAILED: Model config lacks 'patch_size' or processor lacks image size info. Not a standard ViT model.")
            return False
            
        patch_size = config.patch_size
        image_size = processor.crop_size['height'] if hasattr(processor, 'crop_size') else processor.size['height']

        # --- Step 3: Perform Forward Pass ---
        dummy_image = torch.randn(1, 3, image_size, image_size)
        with torch.no_grad():
            outputs = model(dummy_image)

        # --- Step 4: The Core [CLS] Token Checks ---
        
        # Check a) last_hidden_state shape. This is the most crucial check.
        if not hasattr(outputs, 'last_hidden_state'):
            print("❌ FAILED: Model output does not have 'last_hidden_state'.")
            return False
            
        last_hidden_state = outputs.last_hidden_state
        actual_seq_len = last_hidden_state.shape[1]
        num_patches = (image_size // patch_size) ** 2
        expected_seq_len = num_patches + 1

        if actual_seq_len != expected_seq_len:
            print(f"❌ FAILED: Sequence length mismatch.")
            print(f"   - Expected: {expected_seq_len} ({num_patches} patches + 1 [CLS] token)")
            print(f"   - Got: {actual_seq_len}")
            return False
        
        print(f"✅ PASSED: Sequence length is correct ({actual_seq_len}). This confirms a [CLS]-like token exists.")

        # Check b) Existence of pooler_output (only for non-CLIP models)
        if is_clip_model:
            print("ℹ️ INFO: Model is a CLIP variant. 'pooler_output' is not required.")
        elif not hasattr(outputs, 'pooler_output') or outputs.pooler_output is None:
            print("❌ FAILED: Model is not a CLIP model and output does not have 'pooler_output'.")
            return False
        else:
             print(f"✅ PASSED: 'pooler_output' found with shape {outputs.pooler_output.shape}.")

        print(f"➡️ Conclusion: Model '{model_name_or_path}' is COMPATIBLE for token merge.")
        return True

    except Exception as e:
        print(f"❌ FAILED: An error occurred during the check for '{model_name_or_path}'.")
        print(f"   It is likely not a vision model or is otherwise incompatible.")
        # Uncomment the line below for detailed error messages during debugging
        # print(f"   Error details: {e}")
        return False

if __name__ == '__main__':
    # --- Example Usage ---

    # 1. A standard ViT model (should pass)
    check_model_for_token_merge('google/siglip2-so400m-patch14-384')

    # 2. A CLIP vision model (should now pass correctly)
    check_model_for_token_merge('openai/clip-vit-large-patch14')

    # 3. A non-vision, text-based model (should fail)
    check_model_for_token_merge('bert-base-uncased')
    
    # 4. A vision model without a [CLS] token (e.g., SegFormer, should fail)
    check_model_for_token_merge('nvidia/segformer-b0-finetuned-ade-512-512')
