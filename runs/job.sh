#!/bin/bash

python curate.py runs/configs/curation/random-10000-8.yaml
python curate.py runs/configs/curation/random-5000-6.yaml
python curate.py runs/configs/curation/random-1000-4.yaml

python curate.py runs/configs/curation/intra-1000-4.yaml
python curate.py runs/configs/curation/inter-1000-4.yaml
python curate.py runs/configs/curation/inter-intra-1000-4.yaml