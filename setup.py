import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ['absl-py', 'asttokens', 'backcall', 'cachetools', 'certifi', 'charset-normalizer', 'cycler', 'decorator', 'executing', 'fonttools', 'google-auth', 'google-auth-oauthlib', 'grpcio', 'idna', 'ipython', 'jedi', 'kiwisolver', 'Markdown', 'MarkupSafe', 'matplotlib', 'matplotlib-inline', 'numpy', 'oauthlib', 'opencv-python', 'packaging', 'pandas', 'parso', 'pexpect', 'pickleshare', 'Pillow', 'prompt-toolkit', 'protobuf', 'psutil', 'ptyprocess', 'pure-eval', 'pyasn1', 'pyasn1-modules', 'Pygments', 'pyparsing', 'python-dateutil', 'pytz', 'PyYAML', 'requests', 'requests-oauthlib', 'rsa', 'scipy', 'seaborn', 'six', 'stack-data', 'tensorboard', 'tensorboard-data-server', 'tensorboard-plugin-wit', 'thop', 'torch', 'torchvision', 'tqdm', 'traitlets', 'typing_extensions', 'urllib3', 'wcwidth', 'Werkzeug', 'wget']

setuptools.setup(
    name="yolov7_package",
    version="0.0.11",
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
    python_requires='>=3.6',
)
