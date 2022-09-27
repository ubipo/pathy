from setuptools import setup

package_name = 'pathy'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Pieter Fiers',
    maintainer_email='pieter@pfiers.net',
    description='Pathy, the ROS package that runs padnet',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mav = pathy.mav:main',
            'padnet = pathy.padnet:main',
            'camstream = pathy.camstream:main',
            'gcs_webui_ws_server = pathy.gcs_webui_ws_server:main',
            'dms = pathy.dms:main',
            'steering = pathy.steering:main'
        ],
    },
)
