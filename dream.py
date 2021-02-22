import tensorflow as tf
import numpy as np
import matplotlib as matplotlib
import IPython.display as display
import PIL.Image
from tensorflow.keras.preprocessing import image


def download(url, max_dim=None):
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)

def get_local(source, max_dim=None):
    img = PIL.Image.open(source)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)

def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)

def show(img):
    (PIL.Image.fromarray(np.array(img))).show()

def calc_loss(img, model):
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return  tf.reduce_sum(losses)

class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=(tf.TensorSpec(shape=[None,None,3], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.int32), tf.TensorSpec(shape=[], dtype=tf.float32),))
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = calc_loss(img, self.model)
                gradients = tape.gradient(loss, img)
                gradients /= tf.math.reduce_std(gradients) + 1e-8
                img = img + gradients*step_size
                img = tf.clip_by_value(img, -1, 1)

        return loss, img

def run_deep_dream_simple(img, deepdream, steps=100, step_size=0.01):
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining>100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps
        loss, img = deepdream(img, run_steps, tf.constant(step_size))
        display.clear_output(wait=True)
        print("Step {}, loss {}".format(step, loss))

    result = deprocess(img)
    display.clear_output(wait=True)
    show(result)

    return result

def main():
    url = 'https://www.cityam.com/wp-content/uploads/2020/02/London_Tower_Bridge_City.jpg'
    OCTAVE_SCALE = 1.30

    original_img = download(url, max_dim=500) # image from web version
    #original_img = get_local('./eivor.jpeg', max_dim=500) # local image
    show(original_img)

    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    names = ['mixed7', 'mixed8']  # can use 0-10
    target_layers = [base_model.get_layer(name).output for name in names]

    dream_model = tf.keras.Model(inputs=base_model.input, outputs=target_layers)
    deepdream = DeepDream(dream_model)

    img = tf.constant(np.array(original_img))
    base_shape = tf.shape(img)[:-1]
    float_base_shape = tf.cast(base_shape, tf.float32)

    for n in range(-2, 3):
        new_shape = tf.cast(float_base_shape*(OCTAVE_SCALE**n), tf.int32)
        img = tf.image.resize(img, new_shape).numpy()
        img = run_deep_dream_simple(img=img, deepdream=deepdream, steps=50, step_size=0.01)

    display.clear_output(wait=True)
    img = tf.image.resize(img, base_shape)
    img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)
    show(img)

if __name__ == "__main__":
    main()
