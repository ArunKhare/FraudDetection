# ! /bin/bash

find . | grep -E "(__pycache__|\.pyc|\.pyo$|*\.egg*)" | xargs rm -rf