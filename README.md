quasar-decrypt
=======

A spectroscopy analysis tool for processing and analyzing astronomical spectra, 
with focus on emission line fitting and continuum modeling.

Features
--------

- Spectral data loading and preprocessing
- Continuum modeling and fitting
- Emission line detection and measurement
- Balmer series analysis
- Advanced absorption line analysis
- Machine learning-based spectral feature identification

Installation
------------

Install the package in development mode:

.. code-block:: bash

    pip install -e .

Or from the repository:

.. code-block:: bash

    pip install -e /path/to/decrypt

Requirements
^^^^^^^^^^^^

- Python 3.7+
- NumPy
- SciPy
- Pandas
- Matplotlib
- Astropy

Usage
-----

Basic example:

.. code-block:: python

    # Example code usage

Documentation
-------------

See the ``notebooks/`` directory for example notebooks demonstrating various features.

Project Structure
-----------------

::



    decrypt/
    ├── src/decrypt/          # Main package code
    │   ├── continuum/        # Continuum fitting
    │   ├── lines/            # Emission line analysis
    │   ├── balmer/           # Balmer series analysis
    │   ├── absorption/       # Absorption line features
    │   ├── iron/             # Iron multiplet analysis
    │   ├── ml/               # Machine learning models
    │   └── utils/            # Utility functions
    ├── notebooks/            # Jupyter notebooks with examples
    ├── tests/                # Unit tests
    └── scripts/              # Standalone scripts

License
-------

See LICENSE file for details.

Author
------

Liam de Burca <liam_deburca@me.com>
