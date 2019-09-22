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
    SPEED_THRESHOLD = 1.0
    SPEED_THRESHOLD_3 = 3.0
    # Steering penality threshold, change the number based on your action space setting
    ABS_STEERING_THRESHOLD = 20

    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # likely crashed/ close to off track

    if steering > ABS_STEERING_THRESHOLD:
        # Penalize reward if the agent is steering too much
        reward *= 0.8
    elif speed < SPEED_THRESHOLD:
        # Penalize if the car goes too slow
        reward *= 0.1
    else:
        # High reward if the car stays on track and goes fast
        reward *= 1.1
        reward = reward + speed

    reward = reward + (reward * (progress / 100))

    if not all_wheels_on_track:
        # Penalize if the car goes off track
        reward = 1e-3

    return float(reward)