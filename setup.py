from setuptools import setup

setup(
    name='plpy',
    version='0.1',
    description='Utilities for Python program analysis',
    url='https://github.com/josepablocam/python-pl',
    author='Jose Cambronero',
    author_email='jcamsan@mit.edu',
    license='MIT',
    packages=['plpy', 'plpy.analyze', 'plpy.rewrite'],
    install_requires=[
        'astunparse==1.6.3',
        'matplotlib==2.2.2',
        'networkx==2.1',
        'pydot==1.2.4',
        'pandas==0.23.4',
        'numpy==1.22.0',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    zip_safe=False,
)
