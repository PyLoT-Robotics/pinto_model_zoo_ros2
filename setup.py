from setuptools import setup

PACKAGE_NAME = "pinto_model_zoo_ros2"

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=[PACKAGE_NAME],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            f"wholebody_detector = {PACKAGE_NAME}.wholebody_detector_node:main",
        ],
    },
)
