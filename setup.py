from setuptools import setup, find_packages

setup(
	name='Project3',
	version='1.0',
	author='Jwala Katta',
	authour_email='jwala_reddy.katta@ou.edu',
	packages=find_packages(exclude=('tests', 'docs')),
	setup_requires=['pytest-runner'],
	tests_require=['pytest']	
)

