import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as rq:
    requirements = rq.read().splitlines()

setuptools.setup(
    name="yolov7_package",
    version="0.0.1",
    author="Maxim Volkovskiy",
    author_email="maxwolf8852@gmail.com",
    description="Bindings for yolov7 in one class",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maxwolf8852/yolov7_package.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
