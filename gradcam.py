import tensorflow as tf
import numpy as np
import cv2

def generate_gradcam(model, img_array, class_index):
    conv_layer = model.get_layer("block7b_project_conv")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap.numpy(), (300, 300))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = img_array[0]
    img = ((img + 1) * 127.5).astype(np.uint8)

    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
