import setuptools

URL = ""
DESCRIPTION = "Transition state optimizations made easy."
LONG_DESCRIPTION = f"""\
{DESCRIPTION}. For more information, see the [project repository]({URL}).
"""

setuptools.setup(
    name="polanyi",
    version="0.0.1",
    author="Kjell Jorner",
    author_email="kjell.jorner@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    packages=["polanyi"],
    python_requires=">=3.8",
    install_requires=["numpy", "scipy", "wurlitzer"],
    entry_points={
        "console_scripts": [
            "polanyi_xtb_interface=polanyi.xtb_interface:main",
        ]
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
