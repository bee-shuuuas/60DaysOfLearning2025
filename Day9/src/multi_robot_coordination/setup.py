from setuptools import setup
import glob
import os

package_name = 'multi_robot_coordination'

# Define the path to the launch files directory
launch_dir = 'launch/'

# Use glob to get all .launch.py files in the 'launch/' directory
launch_files = glob.glob(os.path.join(launch_dir, '*.launch.py'))

# Define any extra data files (like msg files) here
msg_files = glob.glob('msg/*.msg')  # If you want to install msg files


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', launch_files),  # Include all launch files
        ('share/' + package_name + '/msg', msg_files),  # Include .msg files
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='biswash',
    maintainer_email='your@email.com',
    description='Central multi-robot coordination node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)

