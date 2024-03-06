from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


#----------------------------------Functions needed------------------------------   

def get_attn_last_layer(hidden_features, query_key_weights, query_key_biases):
        """
        This function calculates the attention matrix for the last layer of a Vision Transformer.

        Args:
            hidden_features (torch.Tensor): Hidden features from the model.
            query_key_weights (torch.Tensor): Query key weights for attention calculation.
            query_key_biases (torch.Tensor): Query key biases for attention calculation.

        Returns:
            torch.Tensor: The attention matrix for the last layer.
        """
        
        # Reshape the query key weights and biases
        qkv_w = query_key_weights.reshape(3, -1, 768)  # Shape: (3, num_heads * head_dim, hidden_dim)
        qkv_b = query_key_biases.reshape(3, -1)  # Shape: (3, num_heads * head_dim)

        # Extract query, key, and value weights for 12 heads
        q_12_heads_w = qkv_w[0, :, :]  # Shape: (num_heads * head_dim, hidden_dim)
        k_12_heads_w = qkv_w[1, :, :]  # Shape: (num_heads * head_dim, hidden_dim)
        v_12_heads_w = qkv_w[2, :, :]  # Shape: (num_heads * head_dim, hidden_dim)

        q_12_heads_b = qkv_b[0, :]  # Shape: (num_heads * head_dim)
        k_12_heads_b = qkv_b[1, :]  # Shape: (num_heads * head_dim)
        v_12_heads_b = qkv_b[2, :]  # Shape: (num_heads * head_dim)


        # Calculate Q and K matrices
        Q = torch.matmul(input_features, q_12_heads_w.T) + q_12_heads_b  # Shape: (1, num_tokens, num_heads * head_dim)
        K = torch.matmul(input_features, k_12_heads_w.T) + k_12_heads_b  # Shape: (1, num_tokens, num_heads * head_dim)

        # Calculate the attention matrix
        attn_matrix_last_layer = torch.matmul(Q, K.transpose(2, 1)) / 8  # Shape: (1, num_tokens, num_tokens)
        attn_matrix_last_layer = torch.softmax(attn_matrix_last_layer, dim=(-1))

        return attn_matrix_last_layer

def get_attn_for_heads(hidden_features, query_key_weights, query_key_biases, num_encoder_blocks):
    """
    Calculate attention matrices for each head in the Vision Transformer.

    Args:
        hidden_features (torch.Tensor): Hidden features from the model.
        query_key_weights (torch.Tensor): Query key weights for attention calculation.
        query_key_biases (torch.Tensor): Query key biases for attention calculation.
        num_encoder_blocks (int): Number of encoder blocks.

    Returns:
        List: List of attention matrices for each head.
    """

    # Reshape query key weights and biases
    qkv_w = query_key_weights.reshape(3, -1, 768)  # Shape: (3, num_heads * head_dim, hidden_dim)
    qkv_b = query_key_biases.reshape(3, -1)  # Shape: (3, num_heads * head_dim)

    # Extract query, key, and value weights for 12 heads
    q_12_heads_w = qkv_w[0, :, :]  # Shape: (num_heads * head_dim, hidden_dim)
    k_12_heads_w = qkv_w[1, :, :]  # Shape: (num_heads * head_dim, hidden_dim)
    v_12_heads_w = qkv_w[2, :, :]  # Shape: (num_heads * head_dim, hidden_dim)

    q_12_heads_w = q_12_heads_w.reshape(12, -1, 768)  # Shape: (num_heads, head_dim, hidden_dim)
    k_12_heads_w = k_12_heads_w.reshape(12, -1, 768)  # Shape: (num_heads, head_dim, hidden_dim)
    v_12_heads_w = v_12_heads_w.reshape(12, -1, 768)  # Shape: (num_heads, head_dim, hidden_dim)

    q_12_heads_b = qkv_b[0, :]  # Shape: (num_heads * head_dim)
    k_12_heads_b = qkv_b[1, :]  # Shape: (num_heads * head_dim)
    v_12_heads_b = qkv_b[2, :]  # Shape: (num_heads * head_dim)

    q_12_heads_b = q_12_heads_b.reshape(12, -1)  # Shape: (num_heads,  head_dim)
    k_12_heads_b = k_12_heads_b.reshape(12, -1)  # Shape: (num_heads, head_dim)
    v_12_heads_b = v_12_heads_b.reshape(12, -1)  # Shape: (num_heads, head_dim)

    

    layer_attentions = []

    for i in range(num_encoder_blocks):
        q_w = q_12_heads_w[i, :, :]
        k_w = k_12_heads_w[i, :, :]
        v_w = v_12_heads_w[i, :, :]
        

        q_b = q_12_heads_b[i, :]
        k_b = k_12_heads_b[i, :]
        v_b = v_12_heads_b[i, :]
        

        q = torch.matmul(hidden_features, q_w.T) + q_b
        k = torch.matmul(hidden_features, k_w.T) + k_b
        v = torch.matmul(hidden_features, v_w.T) + v_b
        # print(q.shape, k.shape)
        qk = torch.matmul(q, k.transpose(2, 1)) / 8
        qk = torch.softmax(qk, dim=(-1))
        layer_attentions.append(qk)

    return layer_attentions


def head_operation(layer_attentions, math_operation='mean'):
    """
    Perform an operation on the attention maps from different heads.

    Args:
        layer_attentions (list): List of attention matrices for each head.
        math_operation (str, optional): Math operation to apply ('mean', 'max', 'min'). Defaults to 'mean'.

    Returns:
        torch.Tensor: Final attention map after applying the specified math operation.
    """

    # Concatenate attention matrices from all heads
    layer_attentions_tensor = torch.cat(layer_attentions)

    if math_operation == 'mean':
        # Calculate the mean attention map
        mean_head_attention = torch.mean(layer_attentions_tensor, dim=0)
        final_attention_map = mean_head_attention
    elif math_operation == 'max':
        # Calculate the max attention map
        max_head_attention = torch.max(layer_attentions_tensor, dim=0, keepdim=True)[0][0]
        final_attention_map = max_head_attention
    elif math_operation == 'min':
        # Calculate the min attention map
        min_head_attention = torch.min(layer_attentions_tensor, dim=0, keepdim=True)[0][0]
        final_attention_map = min_head_attention

    return final_attention_map


def plot_attention_per_layer(attn_matrix_last_layer, image_np):
    """
    Plot the attention map overlay on the original image.

    Args:
        attn_matrix_last_layer (torch.Tensor): Attention matrix for the last layer.
        image_np (numpy.ndarray): Original image as a numpy array.
    """

    resized_attention_map = cv2.resize(attn_matrix_last_layer[0, 0, 1:].reshape(14, 14).detach().numpy(), (image_np.shape[1], image_np.shape[0]))

    # Normalize the attention map
    normalized_attention_map = resized_attention_map / np.max(resized_attention_map)

    # Apply the attention map as an overlay on the original image
    result = (normalized_attention_map[..., np.newaxis] * image_np).astype("uint8")

    # Display the original image, attention map, and the overlay
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 20))
    ax1.set_title('Original Image')
    ax2.set_title('Attention Map')
    ax3.set_title('Attention Overlay')
    _ = ax1.imshow(image)
    _ = ax2.imshow(normalized_attention_map, cmap='jet')
    _ = ax3.imshow(result)
    plt.show()


# Define a function to plot attention maps for different heads
def plot_attention_maps_heads(cls_attention_last_layer_all_heads, image, sigma=1):
    """
    Plot attention maps for different heads overlayed on the original image.

    Args:
        cls_attention_last_layer_all_heads (torch.Tensor): Attention maps for different heads.
        image (PIL.Image): Original image.
        sigma (float, optional): Standard deviation for Gaussian filter. Defaults to 1.
    """

    from scipy.ndimage import gaussian_filter, zoom
    import matplotlib.pyplot as plt

    # Apply a Gaussian filter for smoothing
    smoothed_attention_map = gaussian_filter(cls_attention_last_layer_all_heads.reshape(14, 14).detach().numpy(), sigma=sigma)

    # Upscale the image to 256x256 using interpolation
    upscaled_attention_map = zoom(smoothed_attention_map, (image.size[1]/14, image.size[0]/14), order=3)

    # Display the original image with attention map overlay
    plt.imshow(upscaled_attention_map, cmap='jet', alpha=0.7)
    plt.imshow(image, alpha=0.5)
    plt.show()    
#---------------------------------Loading Pre-trained model with DINO weights-------------------------------

vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
num_heads = 12

image = Image.open(r"C:\Users\midok\OneDrive\Desktop\P-51_Mustang_edit1.jpg")

# Define the transformation
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Preprocess the image
input_tensor = preprocess(image)

# Create a mini-batch as expected by the model
input_batch = input_tensor.unsqueeze(0)

features = []


def hook(module, input, output):
    """
    A hook function to capture the output of a specific layer during the forward pass.

    This function is attached to a model layer and is called every time the layer processes input.
    It captures the output of the layer and appends it to a global list `features`.

    Args:
        module (torch.nn.Module): The layer to which this hook is attached.
        input (tuple): The input to the `module`. This is a tuple of tensors representing the inputs to the module.
        output (torch.Tensor): The output from the `module`. This is the tensor produced by the module as output.

    """
    # This is where the output of the layer is captured and appended to the global list `features`.
    features.append(output)

# getting the output from the first normalization layer inside the last encoder block
handle = vitb16.blocks[-1].norm1.register_forward_hook(hook)


vitb16.eval()


# Pass the image through the model at the inference mode
with torch.no_grad():
    outputs = vitb16(input_batch)
    # print(getattr(m.encoder.layers, f'encoder_layer_{i}').self_attention)  
handle.remove()

# print(features[0][0].shape)

vitb16.eval()

with torch.no_grad():
    # We get the output
    outputs = vitb16(input_batch)
    # print(output.shape)

    # So here we get the weights and biases for the quer, key, and value
    qkv_w = vitb16.blocks[-1].attn.qkv.weight
    qkv_b = vitb16.blocks[-1].attn.qkv.bias


    input_features = features[0][0]
    input_features = input_features.unsqueeze(0)
    
    attn_matrix_last_layer = get_attn_last_layer(input_features, qkv_w, qkv_b)

    cls_attention_last_layer = attn_matrix_last_layer[0, 0, 1:]

    image_np = np.array(image)
    plot_attention_per_layer(attn_matrix_last_layer, image_np)

    layer_attentions = get_attn_for_heads(input_features, qkv_w, qkv_b, num_heads)
    final_attention_map = head_operation(layer_attentions)

    cls_attention_last_layer_all_heads = final_attention_map[0, 1:]


    plot_attention_maps_heads(cls_attention_last_layer_all_heads, image, sigma=1)