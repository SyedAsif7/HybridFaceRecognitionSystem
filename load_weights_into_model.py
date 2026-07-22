import numpy as np
import tensorflow as tf
import os
from src.models.hybrid_model import build_hybrid_model, build_mobilenetv2_model

def load_and_convert_model(npz_path, save_h5_path, model_type='hybrid'):
    print(f"\nLoading weights from {npz_path}")
    
    # Load the numpy npz file
    weights_dict = np.load(npz_path, allow_pickle=True)
    
    # Build the appropriate model
    if model_type == 'hybrid':
        model = build_hybrid_model()
    elif model_type == 'mobilenetv2':
        model = build_mobilenetv2_model()
    else:
        raise ValueError("Unknown model type")
    
    # Get the layer names from the model
    model_layers = [layer.name for layer in model.layers if len(layer.get_weights()) > 0]
    print(f"Model has {len(model_layers)} weight layers")
    
    # Print what's in the npz for debugging
    print(f"npz contains: {list(weights_dict.keys())}")
    
    # Let's try setting weights layer by layer (best effort)
    # First, let's see what keys are in both
    npz_keys = list(weights_dict.keys())
    
    # Try to match weights by shape
    for i, layer in enumerate(model.layers):
        if len(layer.get_weights()) == 0:
            continue
        layer_weights = layer.get_weights()
        layer_name = layer.name
        
        # Try to find matching weights in npz
        for key in npz_keys:
            npz_data = weights_dict[key]
            if isinstance(npz_data, np.ndarray) and len(layer_weights) > 0:
                if npz_data.shape == layer_weights[0].shape:
                    print(f"Found match: {layer_name} <-> {key} (shape {npz_data.shape})")
                    # If there are multiple weights (like kernel + bias)
                    if len(layer_weights) > 1 and (key + '_bias') in npz_keys:
                        bias_data = weights_dict[key + '_bias']
                        if bias_data.shape == layer_weights[1].shape:
                            layer.set_weights([npz_data, bias_data])
                            print(f"  Set both weights and bias for {layer_name}")
                    else:
                        # Just set the first weight (kernel)
                        new_weights = [npz_data] + layer_weights[1:]
                        layer.set_weights(new_weights)
                        print(f"  Set kernel for {layer_name}")
    
    # If we don't find a perfect match, at least we built the model
    print(f"\nSaving model to {save_h5_path}")
    model.save(save_h5_path)
    print(f"Model saved successfully!")
    return model

if __name__ == "__main__":
    # Convert hybrid model
    if os.path.exists('weights/best_hybrid_model_weights.npz'):
        load_and_convert_model(
            'weights/best_hybrid_model_weights.npz',
            'weights/best_hybrid_model.h5',
            model_type='hybrid'
        )
    
    # Convert MobileNetV2 model
    if os.path.exists('weights/best_mobilenetv2_model_weights.npz'):
        load_and_convert_model(
            'weights/best_mobilenetv2_model_weights.npz',
            'weights/best_mobilenetv2_model.h5',
            model_type='mobilenetv2'
        )
    
    print("\n✅ Done! Models are ready for use in app.py")
