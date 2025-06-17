import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "pyyaml",
    "opencv-python",
    "torch",
    "torchvision",
    "requests",
    "wget",
    "tqdm",
]

setuptools.setup(
    name="yolov7_package",
    version="0.0.13",
    author="Maxim Volkovskiy",
    author_email="maxwolf8852@gmail.com",
    description="Bindings for yolov7 in one class",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maxwolf8852/yolov7_package.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
