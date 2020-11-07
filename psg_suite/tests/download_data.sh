#!/usr/bin/env sh
cd "$(dirname "$0")"
mkdir data
cp ../sample_data/sample_data.zip data/
(cd data && unzip sample_data.zip)
