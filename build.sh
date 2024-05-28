#!/bin/bash
rm -rf build dist oct_unet.egg-info
python setup.py bdist_wheel --universal
