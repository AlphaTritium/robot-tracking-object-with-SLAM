Here we are trying to simulate a robot trying to track, observe and engage with an object, we have used a cube with a face with 4 purple dots as the target.
First, it tries to find a cube with manual control.
Upon discovery automatically, it will switch to automatic control (unless manual key inputs persists).
First, it will approach and stalk the object.
As it gets close enough it orbits and observed the object, scanning its features, here it scans the faces of the target cube.
When something is identified, which here is the 4 purple dots, it executes the mission of keeping the dots in alignment within the view.
This engagement is done by keeping a distance, whilst tryin to keeping its camera aligned to the centre of the 4 purple dots.

# dedicated launch file



# manual launch

```bash
cd ~/Desktop/robot-tracking-object-with-SLAM
source install/setup.bash

# views
ros2 run rqt_image_view rqt_image_view

# separate manual launches
ros2 launch rc2026_field rc2026_field_sim.launch.py

# enables tracker node
ros2 run robot_tracking_cv tracker_node
ros2 topic echo /tracker/target_pose

# camera view of robot
ros2 run rqt_image_view rqt_image_view
# Select /tracker/debug_view

# launching manual robot control
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# adjusting target distance
ros2 run robot_tracking_cv tracker_node --ros-args -p target_distance:=0.5

# slam module
ros2 launch robot_cartographer cartographer.launch.py
```
