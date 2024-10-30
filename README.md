# Project_AX
- This was the project under subject of robotics in which I have explored usage of ROS2 as well as copelliasim where I simulated Thymio robot for exloration of room and identifing the room based on images taken by thymio robot based in CNN i.e. MobilenetV2 pre trained model which is used to identify rooms by passing images taken and by training them under this model,
- Where, I was able to see that I was getting perfect result of model around 98% accuracy during training also around 71 out of training dataset were truely identified ( True positives ) as you can see in confusion matrix.
- Also, by the end of day I have alos cretedmy own CNN and tried with that which  also came in handy but would like to go wtih pretrinied model for accurate result while realtime detection.
- Done at USI ( Universita Della Svizzeria italiana - Switzerland )
- Live demo is available at https://youtu.be/_-LWYWKpofo?si=ycOfzNEinyq5OUwG


# Datasets 

- Train :
'''
#!/bin/bash
kaggle datasets download romeo62/robotics-1-data-train
'''
