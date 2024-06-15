import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

import time
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.robots import Bimanual
from robosuite.utils.input_utils import *

# from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = choose_environment()

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "bimanual":
            options["robots"] = "Baxter"
        else:
            options["robots"] = []

            # Have user choose two robots
            print("A multiple single-arm configuration was chosen.\n")

            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        options["robots"] = choose_robots(exclude_bimanual=True)

    # Hacky way to grab joint dimension for now
    joint_dim = 6 if options["robots"] == "UR5e" else 7

    # Choose controller
    controller_name = choose_controller()

    # Load the desired controller
    options["controller_configs"] = suite.load_controller_config(
        default_controller=controller_name
    )
    print(options["controller_configs"])
    # import ipdb; ipdb.set_trace()
    # options["controller_configs"]["output_max"] = 10
    # options["controller_configs"]["output_min"] = -10
    # options["controller_configs"]["'damping_ratio"] = 1.0
    # options["controller_configs"]["'impedance_mode"] = "fixed"
    # options["controller_configs"]["kp"] = 150
    # options["controller_configs"]["kp_limits"] = [0, 300]
    # options["controller_configs"]["damping_ratio_limits"] = [0, 10]
    # options["controller_configs"]["input_max"] = 1
    # options["controller_configs"]["input_min"] = -1
    # options["controller_configs"]["velocity_limits"] = [-1, 1]
    # import ipdb; ipdb.set_trace()
    # dataset = RobomimicReplayLowdimDataset(
    #     dataset_path="/media/yaocw/Yiu/diffusion/robomimic/datasets/lift/ph/image.hdf5",
    #     obs_keys=["robot0_joint_vel", "robot0_gripper_qpos"],
    # )
    # import ipdb; ipdb.set_trace()
    import h5py
    with h5py.File("/media/yaocw/Yiu/diffusion/robomimic/datasets/square/ph/image.hdf5") as file:
        demo = file['data']['demo_1']
        obs = demo['obs']
        actions = demo['actions'][:]
        states = demo['states'][:]
        joint_vel = obs['robot0_joint_vel'][:]
    print(f"MAX VEL: {np.max(joint_vel)}")
    # Define the pre-defined controller actions to use (action_dim, num_test_steps, test_value)
    controller_settings = {
        "OSC_POSE": [6, 6, 0.1],
        "OSC_POSITION": [3, 3, 0.1],
        "IK_POSE": [6, 6, 0.01],
        "JOINT_POSITION": [joint_dim, joint_dim, 0.2],
        "JOINT_VELOCITY": [joint_dim, joint_dim, -0.1],
        "JOINT_TORQUE": [joint_dim, joint_dim, 0.25],
    }

    # Define variables for each controller test
    action_dim = controller_settings[controller_name][0]
    num_test_steps = controller_settings[controller_name][1]
    test_value = controller_settings[controller_name][2]

    # Define the number of timesteps to use per controller action as well as timesteps in between actions
    steps_per_action = 75
    steps_per_rest = 75

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        horizon=(steps_per_action + steps_per_rest) * num_test_steps,
        control_freq=20,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    env.sim.set_state_from_flattened(states[1])
    env.sim.forward()

    # To accommodate for multi-arm settings (e.g.: Baxter), we need to make sure to fill any extra action space
    # Get total number of arms being controlled
    n = 0
    gripper_dim = 0
    for robot in env.robots:
        gripper_dim = (
            robot.gripper["right"].dof
            if isinstance(robot, Bimanual)
            else robot.gripper.dof
        )
        n += int(robot.action_dim / (action_dim + gripper_dim))

    # Define neutral value
    neutral = np.zeros(action_dim + gripper_dim)

    # Keep track of done variable to know when to break loop
    count = 0
    # Loop through controller space
    # while count < num_test_steps:
    #     action = neutral.copy()
    #     for i in range(steps_per_action):
    #         if controller_name in {"IK_POSE", "OSC_POSE"} and count > 2:
    #             # Set this value to be the scaled axis angle vector
    #             vec = np.zeros(3)
    #             vec[count - 3] = test_value
    #             action[3:6] = vec
    #         else:
    #             action[count] = test_value
    #         total_action = np.tile(action, n)
    #         env.step(total_action)
    #         env.render()
    #     for i in range(steps_per_rest):
    #         total_action = np.tile(neutral, n)
    #         env.step(total_action)
    #         env.render()
    #     count += 1
    for idx in range(len(actions)):
        action = neutral.copy()
        # action[-1] = 1
        # action[: len(joint_vel[idx])] = joint_vel[idx]
        # action[-1] = actions[idx][-1]
        action = actions[idx]
        env.step(action)
        print(action)
        env.render()

        time.sleep(0.05)

    # Shut down this env before starting the next test
    env.close()
