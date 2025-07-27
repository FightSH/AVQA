
import torch
from .net_encoders import AMS

def run_ams_demo():
    """
    A demo script to initialize and run the AMS model with dummy data.
    """
    # --- 1. Configuration ---
    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Running demo on GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Running demo on CPU.")

    # Model parameters (based on defaults in Net class and AMS class)
    batch_size = 10
    seq_len = 60
    audio_feature_size = 512
    audio_seq_len = 512  # Corresponds to input_size in AMS
    num_nodes = 1       # The feature dimension for the gating network input
    d_model = 512
    d_ff = 64
    num_experts = 4
    k = 2               # Number of experts to select
    patch_size = [15, 10, 6, 3] # Example patch sizes
    
    # The visual-question query feature dimension
    # Based on Transformer_Layer's embeddings_generator: nn.Linear(1536, self.d_model)
    vq_feature_dim = 512

    # --- 2. Model Initialization ---
    print("\nInitializing AMS model...")
    try:
        model = AMS(
            input_size=audio_feature_size,
            output_size=audio_feature_size,
            seq_lenth=60,  # This is the sequence length for the audio input
            num_experts=num_experts,
            device=device,
            d_model=d_model,
            d_ff=d_ff,
            patch_size=patch_size,
            k=k
        ).to(device)
        model.train() # Set to training mode to activate noisy gating
        print("AMS model initialized successfully.")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # --- 3. Prepare Dummy Input Data ---
    # The user suggested a 5D tensor, but based on the AMS.forward and its internal
    # trend_decompose function (which expects a 4D tensor: B, T, D, N),
    # a 4D tensor is the correct input format for 'x'.
    # Shape: (batch_size, time_steps, sequence_length, num_nodes)
    # In the paper's context, this corresponds to (batch, time, features, nodes)
    # Here, we simplify time_steps to 1 for the demo.
    audio_input_shape = (batch_size, seq_len, audio_feature_size)
    dummy_audio_input = torch.randn(audio_input_shape).to(device)

    # The visual-question query tensor 'v_q' is used by the expert's embedding generator.
    # Shape: (batch_size, vq_feature_dim)
    dummy_vq_query = torch.randn(batch_size, 60,vq_feature_dim).to(device)

    print(f"\nCreated dummy audio input 'x' with shape: {dummy_audio_input.shape}")
    print(f"Created dummy visual-question query 'v_q' with shape: {dummy_vq_query.shape}")

    # --- 4. Forward Pass ---
    print("\nPerforming forward pass...")
    try:
        output, balance_loss = model(dummy_audio_input, dummy_vq_query)
        print("Forward pass completed.")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        return

    # --- 5. Print Results ---
    print("\n--- Demo Results ---")
    print(f"Output shape: {output.shape}")
    print(f"Balance loss: {balance_loss.item()}")
    print("--------------------")
    print("\nDemo finished successfully.")


if __name__ == '__main__':
    run_ams_demo()
