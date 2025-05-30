from tensorflow.keras import layers, models, Input

def conv3d_block(input_tensor, n_filters, kernel_size=3):
    """Create a block with two 3D convolutional layers with ReLU activation"""
    x = layers.Conv3D(n_filters, kernel_size, padding='same', activation='relu')(input_tensor)
    x = layers.Conv3D(n_filters, kernel_size, padding='same', activation='relu')(x)
    return x

def encoder_block(input_tensor, n_filters):
    """Create an encoder block with convolution and max pooling"""
    x = conv3d_block(input_tensor, n_filters)
    p = layers.MaxPooling3D(2, padding='same')(x)
    return x, p  # Return both convolution output and pooled output

def decoder_block(input_tensor, skip_features, n_filters, crop_size=((0,0),(0,0),(0,0))):
    """Create a decoder block with transposed convolution and feature concatenation"""
    x = layers.Conv3DTranspose(n_filters, 2, strides=2, padding='same')(input_tensor)
    x = layers.Cropping3D(crop_size)(x)  # Adjust for potential size mismatches
    x = layers.Concatenate()([x, skip_features])  # Skip connection
    x = conv3d_block(x, n_filters)
    return x

def build_model(input_shape_mri=(120, 120, 94, 1), 
                input_shape_template=(120, 120, 94, 4),
                n_classes=4):
    """Build a 3D UNET-like model with dual input pathways"""
    
    # Inputs
    input_mri = Input(input_shape_mri, name='mri_input')
    input_template = Input(input_shape_template, name='template_input')

    # MRI Pathway
    mri_conv1, mri_pool1 = encoder_block(input_mri, 32)    
    mri_conv2, mri_pool2 = encoder_block(mri_pool1, 64)

    # Template Pathway
    temp_conv1, temp_pool1 = encoder_block(input_template, 32)
    temp_conv2, temp_pool2 = encoder_block(temp_pool1, 64)

    # Fusion of both pathways
    merged = layers.Concatenate()([mri_pool2, temp_pool2])

    # Shared Encoder Path
    shared_conv1, shared_pool1 = encoder_block(merged, 128)
    shared_conv2, shared_pool2 = encoder_block(shared_pool1, 256)

    # Bottleneck layer
    bottleneck = conv3d_block(shared_pool2, 512)

    # Decoder Path
    dec1 = decoder_block(bottleneck, shared_conv2, 256, crop_size=((1,0),(1,0),(0,0)))
    dec2 = decoder_block(dec1, shared_conv1, 128)
    
    # Level 3 Upsampling (30x30x24 -> 60x60x47)
    dec3 = layers.Conv3DTranspose(64, 2, strides=2, padding='same')(dec2)
    dec3 = layers.Cropping3D(((0,0), (0,0), (0,1)))(dec3)  # Adjust z-axis dimension
    dec3 = layers.Concatenate()([dec3, mri_conv2, temp_conv2])  # Combine with both pathway features
    dec3 = conv3d_block(dec3, 64)
    
    # Level 4 Upsampling (60x60x47 -> 120x120x94)
    dec4 = layers.Conv3DTranspose(32, 2, strides=2, padding='same')(dec3)
    dec4 = layers.Concatenate()([dec4, mri_conv1, temp_conv1])
    dec4 = conv3d_block(dec4, 32)
    
    # Final output layer
    output = layers.Conv3D(n_classes, 1, activation='softmax')(dec4)
    
    return models.Model(inputs=[input_mri, input_template], outputs=output)