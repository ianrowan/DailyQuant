from setuptools import setup

setup(
    name='DailyQuant',
    version='2.0',
    packages=['tensorflow', 'sklearn', 'objgraph', 'tqdm', 'pandas', 'pandas_datareader', 'matplotlib'],
    url='',
    license='Apache 2.0',
    author='Ian Rowan',
    author_email='ian@MindBuilderAi.com',
    description='DailyQuant utilzies the inherant 3 dimensional structure of stock charts'
                'and their indicies to build a Convolutional Neural Network capable of predicting'
                'the following days projected % gain based of historical quantiles'
)
