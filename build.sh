#!/bin/bash
rm -rf build dist oct_unet.egg-info
python3 setup.py bdist_wheel --universal