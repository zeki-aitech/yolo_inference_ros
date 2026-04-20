from glob import glob
from setuptools import find_packages, setup

package_name = 'yolo_inference_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy>=1.21,<3',
        'torch>=2.0.0',
        'ultralytics>=8.0.0',
    ],
    zip_safe=True,
    maintainer='trungnh',
    maintainer_email='trungnh.aitech@gmail.com',
    description='ROS 2 node for YOLO object detection: 2D/3D detections (vision_msgs), optional tracking and temporal filtering, depth-based 3D bounding boxes.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_inference_node = yolo_inference_ros.yolo_inference_node:main',
        ],
    },
)
