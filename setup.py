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
        'matplotlib==3.2.1',
        'networkx==2.5',
        'pydot==1.4.1',
        'pandas==1.0.3',
        'numpy==1.18.2',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    zip_safe=False,
)
