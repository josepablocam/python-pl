from setuptools import setup

setup(name='plpy',
      version='0.1',
      description='Utilities for Python program analysis',
      url='TODO',
      author='Jose Cambronero',
      author_email='jcamsan@mit.edu',
      license='MIT',
      packages=['plpy'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      zip_safe=False,
      )