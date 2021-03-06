from setuptools import setup

package_name = 'paddy'

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
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mav = paddy.mav:main',
            'padnet = paddy.padnet:main',
            'camstream = paddy.camstream:main',
            'gcs_webui_ws_server = paddy.gcs_webui_ws_server:main',
            'dms = paddy.dms:main',
            'steering = paddy.steering:main'
        ],
    },
)
