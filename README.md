# Context-based gesture control v0.1

## Install

1) Packages install with mamba (recommended):
```
mamba create -n cbgo_env python=3.8
mamba activate cbgo_env
mamba install -c conda-forge -c robostack -c robostack-experimental pymc3 numpy matplotlib pandas pygraphviz seaborn deepdiff scikit-learn arviz aesara ros-noetic-desktop catkin_tools rosdep
```

<details>
<summary>or with Conda:</summary>
<code>conda create -n robot_env python=3.8</code>

<code>conda install -c conda-forge -c robostack -c robostack-experimental pymc3 numpy matplotlib pandas pygraphviz seaborn deepdiff scikit-learn arviz aesara ros-noetic-desktop catkin_tools rosdep</code>
</details>

<details>
<summary>or wih Pip:</summary>

<code>pip install pymc3 numpy matplotlib pandas graphviz seaborn deepdiff scikit-learn arviz aesara</code>

Install ROS noetic manually. Use python version 3.8.
</details>

2) Dependency on [teleop_gesture_toolbox](https://github.com/imitrob/teleop_gesture_toolbox) (ROS, CoppeliaSim, PyRep). Clone also the [ROS interface](https://github.com/imitrob/coppelia_sim_ros_interface) as package.
```
rosdep init
rosdep update
catkin build
```

## Notebooks available

- Run `roscore`
- Run ipykernel (e.g. `jupyter notebook`)

Examples:

- Dataset generator (`nb11_dataset_generation_complete`)
- Mapping gestures to intent, model evaluation (`nb12_model_classification_complete`)
- System pipeline (`nb15_system_pipeline_complete`)

#### Backend test notebooks:

- Robot simulation (`nb10_robot_interface`)
- Behavior tree (`nb14_btree_complete`)
- Scene model introduction (`nb01_scene_and_actions`)
- Scene model moves (`nb04_moves`)
- Scene model additional features (`nb07_scene_additional_features`)
