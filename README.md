# Context-based gesture control v0.1

## Dependencies

- Conda, e.g. Miniconda [download](https://docs.conda.io/en/latest/miniconda.html)
- [Coppelia Sim](https://www.coppeliarobotics.com/) simulator ([install](include/scripts/coppelia_sim_install.sh))
  - (Recommended) Use version 4.1 (PyRep can have problems with newer versions)
  - Please install Coppelia Sim files to your home folder: `~/CoppeliaSim`
```
cd ~
wget --no-check-certificate https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mv CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 CoppeliaSim
rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```

### Optional dependencies:

- [Leap Motion Controller](https://www.ultraleap.com/product/leap-motion-controller/) as a hand sensor ([install](https://developer.leapmotion.com/tracking-software-download), use version 2.3.1)
```
tar -xvzf Leap_Motion_SDK_Linux_2.3.1.tgz
cd LeapDeveloperKit_2.3.1+31549_linux/
sudo dpkg -i Leap-2.3.1+31549-x64.deb
```

## Install

- Packages install with mamba:

```
conda install mamba -c conda-forge # Install mamba

mamba create -n cbgo_env python=3.8
conda activate cbgo_env
mamba env update -n cbgo_env --file environment.yml

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
source $ws/devel/setup.bash
export COPPELIASIM_ROOT=$HOME/CoppeliaSim
export LD_LIBRARY_PATH=$HOME/CoppeliaSim;
export QT_QPA_PLATFORM_PLUGIN_PATH=$HOME/CoppeliaSim;" > ~/activate_cbgo.sh
source ~/activate_cbgo.sh

cd $ws/src
git clone https://github.com/imitrob/PyRep.git
cd PyRep
pip install .
```



## Notebooks available
```
# In terminal 1:
source ~/activate_cbgo.sh
roscore

# in terminal 2:
source ~/activate_cbgo.sh
jupyter-lab
```
#### Examples:

TODO: ADD VISUALS
- Dataset generator (`nb11_dataset_generation_complete`)
- Mapping gestures to intent, model evaluation (`nb12_model_classification_complete`)
- System pipeline (`nb15_system_pipeline_complete`)

#### Backend test notebooks:

- Robot simulation (`nb10_robot_interface`)
- Behavior tree (`nb14_btree_complete`)
- Scene model introduction (`nb01_scene_and_actions`)
- Scene model moves (`nb04_moves`)
- Scene model additional features (`nb07_scene_additional_features`)
