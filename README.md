# Context-based gesture control v0.1

Final version will be available on 2 October 2022.

## Install
Either install with conda:
```
conda install -c conda-forge pymc3 numpy matplotlib pandas pygraphviz seaborn deepdiff scikit-learn arviz aesara
```
or pip:
```
pip install pymc3 numpy matplotlib pandas graphviz seaborn deepdiff scikit-learn arviz aesara
```

Dependency on [teleop_gesture_toolbox](https://github.com/imitrob/teleop_gesture_toolbox) (ROS, CoppeliaSim, PyRep).

## Notebooks available

- Dataset generator (`nb11_dataset_generation_complete`)
- Mapping gestures to intent, model evaluation (`nb12_model_classification_complete`)
- System pipeline (`nb15_system_pipeline_complete` - Full version available on 2 October 22)

#### Backend scene notebooks:

- Robot simulation (`nb10_robot_interface`)
- Behavior tree (`nb14_btree_complete` - Full version available on 2 October 22)
- Scene model introduction (`nb01_scene_and_actions`)
- Scene model moves (`nb04_moves`)
- Scene model additional features (`nb07_scene_additional_features`)
