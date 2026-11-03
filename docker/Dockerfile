FROM ros:humble-ros-base

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pip python3-pytest \
        ros-humble-cv-bridge ros-humble-vision-msgs ros-humble-rmw-fastrtps-cpp ros-humble-rosbag2-storage-mcap ros-humble-rosbag2-py \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# numpy<2: cv_bridge from apt is built against the numpy 1 ABI.
RUN pip3 install --no-cache-dir --timeout 300 --retries 10 "numpy<2" "onnxruntime-gpu[cuda,cudnn]==1.22.0"

# Weights are baked in so the robot needs no internet at the venue.
COPY scripts/download_models.py /usr/local/bin/download_models.py
RUN python3 /usr/local/bin/download_models.py /models

COPY package.xml setup.py setup.cfg /opt/pinto_ws/src/pinto_model_zoo_ros2/
COPY pinto_model_zoo_ros2 /opt/pinto_ws/src/pinto_model_zoo_ros2/pinto_model_zoo_ros2
COPY resource /opt/pinto_ws/src/pinto_model_zoo_ros2/resource
COPY test /opt/pinto_ws/src/pinto_model_zoo_ros2/test
RUN . /opt/ros/humble/setup.sh && cd /opt/pinto_ws && colcon build --symlink-install

COPY docker/fastdds_udp_only.xml /opt/pinto_ws/fastdds_udp_only.xml
COPY docker/entrypoint.sh /entrypoint.sh
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/opt/pinto_ws/fastdds_udp_only.xml
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "run", "pinto_model_zoo_ros2", "wholebody_detector"]
