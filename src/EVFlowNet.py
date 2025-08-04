#!/usr/bin/env python
import math
import tensorflow as tf
import numpy as np

from losses import *
from model import *
from vis_utils import *
from config import configs
from data_loader import get_loader

class EVFlowNet(tf.keras.Model):
    def __init__(self,
                 args,
                 weight_decay_weight=1e-4):
        super().__init__()
        self._args = args
        self._weight_decay_weight = weight_decay_weight
        self.model = EVFlowNetModel(do_batch_norm= True) #not self._args.no_batch_norm)

    def build_graph(self, event_img, prev_img, next_img, is_training=True):
        flow_dict = self.model(event_img,training=is_training ) #not self._args.no_batch_norm)

        wd_loss = tf.add_n([
            tf.nn.l2_loss(v) * self._weight_decay_weight
            for v in self.trainable_variables
        ])

        smoothness_loss = 0
        for i in range(len(flow_dict)):
            smoothness_loss += compute_smoothness_loss(flow_dict[f"flow{i}"])
        smoothness_loss *= self._args.smoothness_weight / 4.

        photometric_loss = compute_photometric_loss(prev_img,
                                                    next_img,
                                                    event_img,
                                                    flow_dict)

        next_image_warped = warp_images_with_flow(next_img, flow_dict['flow3'])

        loss = wd_loss + photometric_loss + smoothness_loss

        return flow_dict, loss, next_image_warped

    def train_model(self, dataset, n_ima):
        print("Starting training.")
        global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self._args.initial_learning_rate,
            decay_steps=int(4. * n_ima / self._args.batch_size),
            decay_rate=self._args.learning_rate_decay,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        summary_writer = tf.summary.create_file_writer(self._args.summary_path)

        debug_rate = 5000
        num_steps = 600000

        @tf.function
        def train_step(event_img, prev_img, next_img):
            with tf.GradientTape() as tape:
                flow_dict, loss, next_image_warped = self.build_graph(event_img, prev_img, next_img, is_training=True)
            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
            global_step.assign_add(1)

            final_flow = flow_dict['flow3']
            event_img_sum = tf.expand_dims(event_img[:, :, :, 0] + event_img[:, :, :, 1], axis=3)
            event_time_img = tf.reduce_max(event_img[:, :, :, 2:4], axis=3, keepdims=True)
            flow_rgb, flow_norm, flow_ang_rad = flow_viz_tf(final_flow)
            image_error = tf.abs(next_image_warped - prev_img)
            image_error = tf.clip_by_value(image_error, 0., 20.)
            color_wheel_rgb = draw_color_wheel_tf(self._args.image_width, self._args.image_height)

            with summary_writer.as_default():
                tf.summary.image("a-Color wheel", color_wheel_rgb, step=global_step)
                tf.summary.image("b-Flow", flow_rgb, step=global_step, max_outputs=self._args.batch_size)
                tf.summary.image("c-Event time image", event_time_img, step=global_step, max_outputs=self._args.batch_size)
                tf.summary.image('d-Warped_next_image', next_image_warped, step=global_step, max_outputs=self._args.batch_size)
                tf.summary.image("e-Prev image", prev_img, step=global_step, max_outputs=self._args.batch_size)
                tf.summary.image("f-Image error", image_error, step=global_step, max_outputs=self._args.batch_size)
                tf.summary.image("g-Event image", event_img_sum, step=global_step, max_outputs=self._args.batch_size)
            return loss

        for step, batch in enumerate(dataset):
            if step >= num_steps:
                break
            event_img, prev_img, next_img, _ = batch  # Assuming dataset yields these
            loss_value = train_step(event_img, prev_img, next_img)
            if step % debug_rate == 0:
                print(f"Step {step}, Loss: {loss_value.numpy()}")

if __name__ == "__main__":
    # Example main for TF2
    # You need to implement configs() and get_loader() as in your repo
    args = configs()
    dataset, n_ima = get_loader(
        args.data_path,
        args.batch_size,
        args.image_width,
        args.image_height,
        split='train',
        shuffle=True,
        sequence=args.train_sequence,
        skip_frames=args.train_skip_frames,
        time_only=args.time_only,
        count_only=args.count_only
    )
    evflownet = EVFlowNet(args)
    evflownet.train_model(dataset, n_ima)