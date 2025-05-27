from tensorflow.keras import layers, models, Input

def conv3d_block(input_tensor, n_filters, kernel_size=3):
    x = layers.Conv3D(n_filters, kernel_size, padding='same', activation='relu')(input_tensor)
    x = layers.Conv3D(n_filters, kernel_size, padding='same', activation='relu')(x)
    return x

def encoder_block(input_tensor, n_filters):
    x = conv3d_block(input_tensor, n_filters)
    p = layers.MaxPooling3D(2, padding='same')(x)
    return x, p

def decoder_block(input_tensor, skip_features, n_filters, crop_size=((0,0),(0,0),(0,0))):
    x = layers.Conv3DTranspose(n_filters, 2, strides=2, padding='same')(input_tensor)
    x = layers.Cropping3D(crop_size)(x)
    x = layers.Concatenate()([x, skip_features])
    x = conv3d_block(x, n_filters)
    return x

def build_model(input_shape=(120, 120, 94, 1), n_classes=4):
    input_mri = Input(input_shape, name='mri_input')
    
    # MRI Pathway
    mri_conv1, mri_pool1 = encoder_block(input_mri, 32)
    mri_conv2, mri_pool2 = encoder_block(mri_pool1, 64)
    
    # Shared encoder
    shared_conv1, shared_pool1 = encoder_block(mri_pool2, 128)
    shared_conv2, shared_pool2 = encoder_block(shared_pool1, 256)

    # Bottleneck
    bottleneck = conv3d_block(shared_pool2, 512)

    # Decoder
    dec1 = decoder_block(bottleneck, shared_conv2, 256, crop_size=((1,0),(1,0),(0,0)))
    dec2 = decoder_block(dec1, shared_conv1, 128)
    dec3 = layers.Conv3DTranspose(64, 2, strides=2, padding='same')(dec2)
    dec3 = layers.Cropping3D(((0,0), (0,0), (0,1)))(dec3)
    dec3 = layers.Concatenate()([dec3, mri_conv2])
    dec3 = conv3d_block(dec3, 64)
    dec4 = layers.Conv3DTranspose(32, 2, strides=2, padding='same')(dec3)
    dec4 = layers.Concatenate()([dec4, mri_conv1])
    dec4 = conv3d_block(dec4, 32)
    
    output = layers.Conv3D(n_classes, 1, activation='softmax')(dec4)
    
    return models.Model(inputs=input_mri, outputs=output)