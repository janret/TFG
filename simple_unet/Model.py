from tensorflow.keras import layers, models, Input

def conv3d_block(input_tensor, n_filters, kernel_size=3):
    conv = layers.Conv3D(n_filters, kernel_size, padding='same', activation='relu')(input_tensor)
    conv = layers.Conv3D(n_filters, kernel_size, padding='same', activation='relu')(conv)
    return conv

def encoder_block(input_tensor, n_filters):
    conv_features = conv3d_block(input_tensor, n_filters)
    pooled_features = layers.MaxPooling3D(2, padding='same')(conv_features)
    return conv_features, pooled_features

def decoder_block(input_tensor, skip_features, n_filters, crop_size=((0,0),(0,0),(0,0))):
    upsampled = layers.Conv3DTranspose(n_filters, 2, strides=2, padding='same')(input_tensor)
    cropped_upsampled = layers.Cropping3D(crop_size)(upsampled)
    merged = layers.Concatenate()([cropped_upsampled, skip_features])
    conv_output = conv3d_block(merged, n_filters)
    return conv_output

def build_model(input_shape=(120, 120, 94, 1), n_classes=4):
    input_mri = Input(input_shape, name='mri_input')
    
    # MRI Encoder Path
    enc1_features, enc1_pooled = encoder_block(input_mri, 32)
    enc2_features, enc2_pooled = encoder_block(enc1_pooled, 64)
    
    # Shared deeper encoder path
    enc3_features, enc3_pooled = encoder_block(enc2_pooled, 128)
    enc4_features, enc4_pooled = encoder_block(enc3_pooled, 256)

    # Bottleneck
    bottleneck_features = conv3d_block(enc4_pooled, 512)

    # Decoder Path
    decoder4 = decoder_block(bottleneck_features, enc4_features, 256, crop_size=((1,0),(1,0),(0,0)))
    decoder3 = decoder_block(decoder4, enc3_features, 128)
    
    # Special handling for decoder2 due to asymmetric cropping
    decoder2_upsampled = layers.Conv3DTranspose(64, 2, strides=2, padding='same')(decoder3)
    decoder2_cropped = layers.Cropping3D(((0,0), (0,0), (0,1)))(decoder2_upsampled)
    decoder2_merged = layers.Concatenate()([decoder2_cropped, enc2_features])
    decoder2 = conv3d_block(decoder2_merged, 64)
    
    # Final decoder block
    decoder1_upsampled = layers.Conv3DTranspose(32, 2, strides=2, padding='same')(decoder2)
    decoder1_merged = layers.Concatenate()([decoder1_upsampled, enc1_features])
    decoder1 = conv3d_block(decoder1_merged, 32)
    
    output_segmentation = layers.Conv3D(n_classes, 1, activation='softmax')(decoder1)
    
    return models.Model(inputs=input_mri, outputs=output_segmentation)