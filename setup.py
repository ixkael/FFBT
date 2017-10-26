from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
from sphinx.setup_command import BuildDoc

version = '1.0.0'

args = {
    "libraries": ["m"],
    "include_dirs": [numpy.get_include()],
    "extra_link_args": ['-fopenmp'],
    "extra_compile_args": ["-ffast-math", "-fopenmp",
                           "-Wno-uninitialized",
                           "-Wno-maybe-uninitialized",
                           "-Wno-unused-function"]  # -march=native
    }

ext_modules = [
    Extension("ffbt.utils_cy",
              ["ffbt/utils_cy.pyx"], **args)
    ]

setup(
  name="ffbt",
  version=version,
  cmdclass={"build_ext": build_ext,
            'build_sphinx': BuildDoc},
  packages=['ffbt'],
  command_options={
        'build_sphinx': {
            'project': (None, "ffbt"),
            'version': ('setup.py', version),
            'build_dir': (None, 'docs/_build'),
            'config_dir': (None, 'docs'),
            }},
  ext_modules=ext_modules)
