#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, print_function

from os.path import basename, dirname, join, splitext

import io
import os
from glob import glob
from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
# torch extensions
from torch.utils.cpp_extension import BuildExtension, CppExtension

try:
    # Allow installing package without any Cython available. This
    # assumes you are going to include the .c files in your sdist.
    import Cython
except ImportError:
    Cython = None

# Enable code coverage for C code: we can't use CFLAGS=-coverage in tox.ini, since that may mess with compiling
# dependencies (e.g. numpy). Therefore we set SETUPPY_CFLAGS=-coverage in tox.ini and copy it to CFLAGS here (after
# deps have been safely installed).
if 'TOX_ENV_NAME' in os.environ and os.environ.get('SETUP_PY_EXT_COVERAGE') == 'yes':
    CFLAGS = os.environ['CFLAGS'] = '-DCYTHON_TRACE=1'
    LFLAGS = os.environ['LFLAGS'] = ''
else:
    CFLAGS = ''
    LFLAGS = ''


class optional_build_ext(build_ext):
    """Allow the building of C extensions to fail."""
    def run(self):
        try:
            build_ext.run(self)
        except Exception as e:
            self._unavailable(e)
            self.extensions = []  # avoid copying missing files (it would fail).

    def _unavailable(self, e):
        print('*' * 80)
        print('''WARNING:

    An optional code optimization (C extension) could not be compiled.

    Optimizations for this package will not be available!
        ''')

        print('CAUSE:')
        print('')
        print('    ' + repr(e))
        print('*' * 80)


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name='casper3d',
    use_scm_version={
        'local_scheme': 'dirty-tag',
        'write_to': 'src/casper3d/_version.py',
        'fallback_version': '0.1.0',
    },
    license='MIT',
    description='Category-level Pose Estimation',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    install_requires=[
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
        'torch'
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    setup_requires=[
        'setuptools_scm>=3.3.1',
        'cython',
        'torch'
    ] if Cython else [
        'setuptools_scm>=3.3.1',
        'torch'
    ],
    ext_modules=[
        #Extension(
        #    splitext(relpath(path, 'src').replace(os.sep, '.'))[0],
        #    sources=[path],
        #    extra_compile_args=CFLAGS.split(),
        #    extra_link_args=LFLAGS.split(),
        #    include_dirs=[dirname(path)]
        #)
        #for root, _, _ in os.walk('src')
        #for path in glob(join(root, '*.pyx' if Cython else '*.c'))
        #CUDAExtension('trilinear_interpolate_cuda', [
        #    'src/utils/kernels/trilinear_interpolate_cuda.cpp',
        #    'src/utils/kernels/trilinear_interpolate_cuda_kernel.cu',
        #]),
        CppExtension(
            name='cosypose_cext',
            sources=[
                'src/cosypose/csrc/cosypose_cext.cpp'
            ],
            extra_compile_args=['-O3'],
            verbose=True
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
