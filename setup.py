from setuptools import setup, find_packages

setup(
  name="deepod-snt",
  version="0.1.0",
  author="Yann HOFFMANN",
  author_email="yann.hoffmann@uni.lu",
  description="A package for deep outlier detection with constraints and visualization.",
  long_description=open("README.md").read(),
  long_description_content_type="text/markdown",
  url="https://github.com/Yann21/deepod-snt",  # Replace with the actual repository URL
  packages=find_packages(),
  include_package_data=True,
  install_requires=open("requirements.txt").read().splitlines(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  python_requires=">=3.6",
  entry_points={
    # "console_scripts": [
      # "deepod-snt=deepod-snt.main:main",  # Ensure `main.py` has a `main()` function
    # ],
  },
)
