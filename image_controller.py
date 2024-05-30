import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import os

import torch
from torchvision import transforms
from PIL import Image as Imim

from thymiroomba.controller import ControllerNode
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from copy import deepcopy
from enum import Enum
from math import inf
import random
import sys
import time 
from datetime import datetime 


class ThymioState(Enum):
    FORWARD = 1
    BACKUP = 2
    ROTATING = 3


class ImageController(ControllerNode):
    UPDATE_STEP = 1 / 20
    OUT_OF_RANGE = 1
    TARGET_DISTANCE = OUT_OF_RANGE - 0.04
    TOO_CLOSE = 0.05
    TARGET_ERROR = 0.001

    

    def __init__(self):
        super().__init__('image_controller', update_step=self.UPDATE_STEP)
        
        self.model = None
        self.filename = None
        
        self.preproc = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(240),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.current_state = None
        self.next_state = ThymioState.FORWARD
        self.image = None
        self.odom_pose = None
        self.bridge = CvBridge()
        self.next_run_time = time.time()

        self.COLORS = {0: [0.0, 0.0, 1.0], 1: [0.0, 1.0, 0.0]}

        self.front_sensors = ["center_left", "center", "center_right"]
        self.lateral_sensors = ["left", "right"]
        self.rear_sensors = ["rear_left", "rear_right"]
        self.proximity_sensors = self.front_sensors + self.lateral_sensors + self.rear_sensors
        self.proximity_distances = dict()
        self.proximity_subscribers = [
            self.create_subscription(Range, f'proximity/{sensor}', self.create_proximity_callback(sensor), 10)
            for sensor in self.proximity_sensors
        ]

        self.image_subscriber = self.create_subscription(Image, '/thymio0/camera', self.image_reader, 1)
        
        self.image_save_directory = "/home/robotics23/dev_ws/src/robotics_project/robotics_project/Room_3"
        
        self.led_publisher_left = self.create_publisher(ColorRGBA,'/thymio0/led/body/bottom_left',10)
        self.led_publisher_right =self.create_publisher(ColorRGBA,'/thymio0/led/body/bottom_right',10)
        self.led_publisher_top =self.create_publisher(ColorRGBA,'/thymio0/led/body/top',10)

        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.loadModel()

    def image_reader(self, msg):
        self.get_logger().info("Entering image cache")
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if image is not None:
                self.image = image
                self.get_logger().info("Image received and decoded successfully")
            else:
                self.get_logger().warn("Failed to decode image")
        except Exception as e:
            self.get_logger().error(f"Exception in image_reader: {e}")

    def create_proximity_callback(self, sensor):
        def proximity_callback(msg):
            self.proximity_distances[sensor] = msg.range if msg.range >= 0.0 else inf
            self.get_logger().debug(
                f"proximity: {self.proximity_distances}",
                throttle_duration_sec=0.5
            )

        return proximity_callback

    def save_image(self):
        current_time = time.time()
        if current_time >= self.next_run_time:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.filename = os.path.join(self.image_save_directory, f"image_{current_time}.jpg")
            
            if self.image is not None:
                cv2.imwrite(self.filename, self.image)
                self.get_logger().info(f"Image saved to: {self.filename}")
            else:
                self.get_logger().warn("No image to save")
            self.next_run_time = time.time() + 5
        self.predict_image()     


    def turn_on_led(self, color):
        rgba = list(color) + [1.0]
        self.get_logger().info(f"{rgba} is shown")

        color_msg = ColorRGBA()
        color_msg.r = rgba[0]
        color_msg.g = rgba[1]
        color_msg.b = rgba[2]
        color_msg.a = rgba[3]

        self.led_publisher_left.publish(color_msg)
        self.led_publisher_right.publish(color_msg)
        self.led_publisher_top.publish(color_msg)

    def loadModel(self):
            #Our mobilenetv2 model will go here. CNN is too large
        model_path = "/home/robotics/dev_ws/src/robotics_project/robotics_project/MobileNetmodel.pth"
        self.model = torch.load(model_path,map_location=torch.device('cpu'))
        self.model.eval()

    def predict_image(self):
        self.get_logger().info("predicting image")
        if self.model is None:
            self.loadModel()
        if self.model is not None:
            self.get_logger().info("model loaded confirmed")
            if self.filename is not None:
                self.get_logger().info("Inside prediction chain")
            
                image = Imim.open(self.filename).convert('RGB')  # Ensure the image is in RGB mode
                image = self.preproc(image).unsqueeze(0).to("cpu")
                
                with torch.no_grad():
                    output = self.model(image)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    _, predicted = torch.max(output, 1)
                
                rnnum= predicted.item()
                self.get_logger().info(f"{rnnum} is type {type(rnnum)}")
                
                probVal = probabilities.cpu().numpy()[0][rnnum]
                self.get_logger().info(f"{probVal} is probability of room {rnnum +1}")

                if rnnum==0:
                    if probVal>0.5:
                        self.get_logger().info("In room 1")
                        self.turn_on_led(self.COLORS[rnnum])
                else:
                    if probVal>0.5:
                        self.get_logger().info("In room 2")
                        self.turn_on_led(self.COLORS[rnnum])
                    
                    
      

    def update_callback(self):
        if self.odom_pose is None or len(self.proximity_distances) < len(self.proximity_sensors):
            return

        if self.next_state != self.current_state:
            self.get_logger().info(f"state_machine: transitioning from {self.current_state} to {self.next_state}")

            if self.next_state == ThymioState.FORWARD:
                self.init_forward()
            elif self.next_state == ThymioState.BACKUP:
                self.init_backup()
            elif self.next_state == ThymioState.ROTATING:
                self.init_rotating()

            self.current_state = self.next_state
        self.save_image()

        if self.current_state == ThymioState.FORWARD:
            self.update_forward()
        elif self.current_state == ThymioState.BACKUP:
            self.update_backup()
        elif self.current_state == ThymioState.ROTATING:
            self.update_rotating()

    def init_forward(self):
        self.stop()

    def update_forward(self):
        if any(self.proximity_distances[sensor] < self.TARGET_DISTANCE for sensor in self.front_sensors):
            self.next_state = ThymioState.BACKUP
            return

        cmd_vel = Twist()
        cmd_vel.linear.x = 1.5
        cmd_vel.angular.z = 0.0
        self.vel_publisher.publish(cmd_vel)
        self.get_logger().info("Moving forward with constant velocity")

    def init_backup(self):
        self.stop()
    
    def update_backup(self):
        self.get_logger().info("Hit the wall")
        if all(self.proximity_distances[sensor] > self.TOO_CLOSE for sensor in self.front_sensors):
            self.next_state = ThymioState.ROTATING
            return

        cmd_vel = Twist()
        cmd_vel.linear.x = -1.5
        cmd_vel.angular.z = 0.0
        self.vel_publisher.publish(cmd_vel)

    def init_rotating(self):
        self.stop()
        self.turn_direction = random.choice([-1, 1])

    def update_rotating(self):
        if all(self.proximity_distances[sensor] == inf for sensor in self.front_sensors):
            self.next_state = ThymioState.FORWARD
            return

        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = self.turn_direction * 3.0
        self.vel_publisher.publish(cmd_vel)


def main():
    rclpy.init(args=sys.argv)
    node = ImageController()
    node.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.stop()


if __name__ == '__main__':
    main()

