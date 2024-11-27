import setuptools

setuptools.setup(
      include_package_data = True,
      name         = 'Vibrations2D',
      version      = '1.0',
      description  = 'Python tools for 2d IR spectroscopy',
      author       = 'Christoph Jacob, Julia Brueggemann, Mario Wolter, Michael Welzel,\
                      Maria Chekmeneva, Anna Maria van Bodegraven and others',
      url          = 'https://www.tu-braunschweig.de/pci/agjacob/software',
      license      = 'GPLv3',
      package_dir  = {'': 'src/'},
      python_requires = '>=3.11.4',
      install_requires = ['numpy>=1.23.4','scipy>=1.9.3'],
      classifiers  = ["Programming Language :: Python :: 3",
                      "Operating System :: OS Independent"],
     )
