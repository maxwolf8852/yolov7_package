import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ['absl-py>=1.2.0', 'asttokens>=2.0.8', 'backcall>=0.2.0', 'cachetools>=5.2.0', 'certifi>=2022.6.15', 'charset-normalizer>=2.1.0', 'cycler>=0.11.0', 'decorator>=5.1.1', 'executing>=0.10.0', 'fonttools>=4.35.0', 'google-auth>=2.10.0', 'google-auth-oauthlib>=0.4.6', 'grpcio>=1.47.0', 'idna>=3.3', 'ipython>=8.4.0', 'jedi>=0.18.1', 'kiwisolver>=1.4.4', 'Markdown>=3.4.1', 'MarkupSafe>=2.1.1', 'matplotlib>=3.5.3', 'matplotlib-inline>=0.1.5', 'numpy>=1.23.2', 'oauthlib>=3.2.0', 'opencv-python>=4.6.0.66', 'packaging>=21.3', 'pandas>=1.4.3', 'parso>=0.8.3', 'pexpect>=4.8.0', 'pickleshare>=0.7.5', 'Pillow>=9.2.0', 'prompt-toolkit>=3.0.30', 'protobuf>=3.19.4', 'psutil>=5.9.1', 'ptyprocess>=0.7.0', 'pure-eval>=0.2.2', 'pyasn1>=0.4.8', 'pyasn1-modules>=0.2.8', 'Pygments>=2.13.0', 'pyparsing>=3.0.9', 'python-dateutil>=2.8.2', 'pytz>=2022.2.1', 'PyYAML>=6.0', 'requests>=2.28.1', 'requests-oauthlib>=1.3.1', 'rsa>=4.9', 'scipy>=1.9.0', 'seaborn>=0.11.2', 'six>=1.16.0', 'stack-data>=0.4.0', 'tensorboard>=2.10.0', 'tensorboard-data-server>=0.6.1', 'tensorboard-plugin-wit>=1.8.1', 'thop>=0.1.1.post2207130030', 'torch>=1.12.1', 'torchvision>=0.13.1', 'tqdm>=4.64.0', 'traitlets>=5.3.0', 'typing_extensions>=4.3.0', 'urllib3>=1.26.11', 'wcwidth>=0.2.5', 'Werkzeug>=2.2.2', 'wget>=3.2']

setuptools.setup(
    name="yolov7_package",
    version="0.0.5",
    author="Maxim Volkovskiy",
    author_email="maxwolf8852@gmail.com",
    description="Bindings for yolov7 in one class",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maxwolf8852/yolov7_package.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
