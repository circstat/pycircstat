from setuptools import setup

setup(
    name = "PyCircStat",
    version = "0.0.1",
    author = "Philipp Berens et al.",
    author_email = "philipp.berens@uni-tuebingen.de",
    description = ("Toolbox for circular statistics with Python"),
    #license = "MIT",
    keywords = "statistics",
    #url = "http://packages.python.org/PyCircStat",
    packages=['pycircstat', 'tests'],
    #long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        #"License :: OSI Approved :: MIT License",
    ],
    setup_requires=['nose>=1.0', 'mock', 'sphinx_rtd_theme'],
)
