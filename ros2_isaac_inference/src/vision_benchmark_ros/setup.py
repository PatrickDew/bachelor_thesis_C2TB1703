import os
from glob import glob

from setuptools import find_packages, setup

package_name = "vision_benchmark_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "isaac_multitask_node = vision_benchmark_ros.isaac_multitask_node:main",
        ],
    },
    maintainer="vision_benchmark",
    maintainer_email="user@example.com",
    description="Isaac Sim image subscriber with pose and instance segmentation.",
    license="Apache-2.0",
    tests_require=["pytest"],
)
