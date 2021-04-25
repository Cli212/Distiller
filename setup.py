from setuptools import setup, find_packages

setup(
    name = "distiller",
    version = "1.0",
    keywords = ("distiller", "knowledge distillation"),
    description = "eds sdk",
    long_description = "eds sdk for python",
    license = "MIT Licence",

    url = "http://test.com",
    author = "test",
    author_email = "test@gmail.com",

    packages=find_packages("src"),
    package_dir={'': 'src'},
    include_package_data = True,
    platforms = "any",
    install_requires = [],

    scripts = [],
    entry_points = {
        'console_scripts': [
            'test = test.help:main'
        ]
    }
)