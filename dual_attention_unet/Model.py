from tensorflow.keras import layers, models, Input
import tensorflow as tf

def attention_gate(x, g, filters):
    """Attention gate to focus on relevant features"""
    theta_x = layers.Conv3D(filters, 1, padding='same')(x)
    phi_g = layers.Conv3D(filters, 1, padding='same')(g)
    
    f = layers.Activation('relu')(layers.Add()([theta_x, phi_g]))
    psi_f = layers.Conv3D(1, 1, padding='same')(f)
    
    rate = layers.Activation('sigmoid')(psi_f)
    att_x = layers.Multiply()([x, rate])
    
    return att_x

def conv3d_block(input_tensor, n_filters, kernel_size=3):
    """Create a block with two 3D convolutional layers with ReLU activation"""
    x = layers.Conv3D(n_filters, kernel_size, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv3D(n_filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def encoder_block(input_tensor, n_filters):
    """Create an encoder block with convolution and max pooling"""
    x = conv3d_block(input_tensor, n_filters)
    p = layers.MaxPooling3D(2, padding='same')(x)
    return x, p

def decoder_block(input_tensor, skip_features, n_filters, crop_size=((0,0),(0,0),(0,0))):
    """Create a decoder block with transposed convolution and feature concatenation"""
    x = layers.Conv3DTranspose(n_filters, 2, strides=2, padding='same')(input_tensor)
    x = layers.Cropping3D(crop_size)(x)
    x = layers.Concatenate()([x, skip_features])
    x = conv3d_block(x, n_filters)
    return x

def template_processing_block(template_input, n_filters):
    """Process template input (4 segmentation channels)"""
    # Process all segmentation channels together
    template_feat = conv3d_block(template_input, n_filters)
    
    # Add self-attention to enhance feature relationships between different segmentation classes
    attention_feat = attention_gate(template_feat, template_feat, n_filters)
    
    # Final processing
    return conv3d_block(attention_feat, n_filters)

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
    template_processed = template_processing_block(input_template, 32)
    temp_conv1, temp_pool1 = encoder_block(template_processed, 32)
    temp_conv2, temp_pool2 = encoder_block(temp_pool1, 64)

    # Fusion of pathways with attention
    mri_att = attention_gate(mri_pool2, temp_pool2, 64)
    temp_att = attention_gate(temp_pool2, mri_pool2, 64)
    merged = layers.Concatenate()([mri_att, temp_att])

    # Shared Encoder Path
    shared_conv1, shared_pool1 = encoder_block(merged, 128)
    shared_conv2, shared_pool2 = encoder_block(shared_pool1, 256)

    # Bottleneck with dropout for regularization
    bottleneck = conv3d_block(shared_pool2, 512)
    bottleneck = layers.Dropout(0.3)(bottleneck)

    # Decoder Path with attention gates
    dec1 = decoder_block(bottleneck, shared_conv2, 256, crop_size=((1,0),(1,0),(0,0)))
    dec2 = decoder_block(dec1, shared_conv1, 128)
    
    # Level 3 Upsampling with attention
    dec3 = layers.Conv3DTranspose(64, 2, strides=2, padding='same')(dec2)
    dec3 = layers.Cropping3D(((0,0), (0,0), (0,1)))(dec3)
    dec3_att = attention_gate(dec3, layers.Concatenate()([mri_conv2, temp_conv2]), 64)
    dec3 = layers.Concatenate()([dec3_att, mri_conv2, temp_conv2])
    dec3 = conv3d_block(dec3, 64)
    
    # Level 4 Upsampling with attention
    dec4 = layers.Conv3DTranspose(32, 2, strides=2, padding='same')(dec3)
    dec4_att = attention_gate(dec4, layers.Concatenate()([mri_conv1, temp_conv1]), 32)
    dec4 = layers.Concatenate()([dec4_att, mri_conv1, temp_conv1])
    dec4 = conv3d_block(dec4, 32)
    
    # Final output layer
    output = layers.Conv3D(n_classes, 1, activation='softmax')(dec4)
    
    return models.Model(
        inputs={
            'mri_input': input_mri,
            'template_input': input_template
        },
        outputs=output
    ) 