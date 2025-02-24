# vision_based_driving_DL_HW5
CNN model to do vision-based driving in SuperTuxKart
design a simple low-level controller that acts as an auto-pilot to drive in SuperTuxKart. We then use this auto-pilot to train a vision based driving system. 

# Controller
In the first part of this homework, you will write a low-level controller in controller.py. The controller function takes as input an aim point and the current velocity of the car. The aim point is a point on the center of the track 15 meters away from the kart

# Planner
In the second part, you will train a planner to predict the aim point. The planner takes as input an image and outputs the aim point in the image coordinate. Your controller then maps those aim points to actions.
