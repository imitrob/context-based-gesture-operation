# Context-based gesture control v0.1

## Install

- Packages install with mamba (recommended):
- Dependency on [teleop_gesture_toolbox](https://github.com/imitrob/teleop_gesture_toolbox) (ROS, CoppeliaSim, PyRep) and [ROS interface](https://github.com/imitrob/coppelia_sim_ros_interface).
```
mamba create -n cbgo_env python=3.8
conda activate cbgo_env
mamba install -c conda-forge -c robostack -c robostack-experimental pymc3 numpy matplotlib pandas pygraphviz seaborn deepdiff scikit-learn arviz aesara ros-noetic-desktop ros-noetic-moveit-visual-tools catkin_tools rosdep 

# Reactivate conda env before proceeding. 
conda deactivate
conda activate cbgo_env  

export ws=<path/to/catkin/ws>
mkdir -p $ws/src
cd $ws/src
git clone https://github.com/imitrob/context_based_gesture_operation.git
git clone https://github.com/imitrob/teleop_gesture_toolbox.git
git clone https://github.com/imitrob/coppelia_sim_ros_interface.git

cd $ws
rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -r -y
catkin build

source $ws/devel/setup.bash

# make activation script
echo "export ws=$ws
conda activate cbgo_env
source $ws/devel/setup.bash" > ~/activate_cbgo.sh
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




## Notebooks available
```
# In terminal 1:
source ~/activate_cbgo.sh
roscore

# in terminal 2:
source ~/activate_cbgo.sh
jupyter-lab
```
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
