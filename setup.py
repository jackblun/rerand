from distutils.core import setup

setup(
    name="rerand",
    version="0.1.0",
    author="Jack Blundell",
    author_email="jackblun@gmail.com",
    packages=["rerand"],
    url="http://pypi.python.org/pypi/rerand/",
    license="LICENSE.txt",
    description="Tools for rerandomisation in randomised experiments.",
    long_description=open("README.txt").read(),
    install_requires=["numpy >= 1.1.1"],
)
