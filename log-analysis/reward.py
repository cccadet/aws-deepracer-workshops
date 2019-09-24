def reward_function(params):
    '''
    Example of rewarding the agent to follow center line

    all_wheels_on_track (True|False):
    distance_from_center (0:~track_width/2):
    progress (0:100):
    steering_angle (-30:30):
    speed (0.0:5.0):

    '''

    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    all_wheels_on_track = params['all_wheels_on_track']
    steering = abs(params['steering_angle'])  # Only need the absolute steering angle
    progress = params['progress']
    speed = params['speed']
    # Steering penality threshold, change the number based on your action space setting
    ABS_STEERING_THRESHOLD = 20

    if not all_wheels_on_track:
        # Penalize if the car goes off track
        reward = 1e-3
    else:
        reward = speed

    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.4 * track_width
    marker_2 = 0.45 * track_width
    marker_3 = 0.5 * track_width

    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward *= 1
    elif distance_from_center <= marker_2:
        reward *= 0.9
    elif distance_from_center <= marker_3:
        reward *= 0.85
    else:
        reward = 1e-3  # likely crashed/ close to off track

    if steering > ABS_STEERING_THRESHOLD:
        # Penalize reward if the agent is steering too much
        reward *= 0.8

    reward = reward + (reward * (progress / 100))


    return float(reward)