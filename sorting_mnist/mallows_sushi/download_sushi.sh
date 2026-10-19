#!/bin/bash
# Download the Sushi preference data (Kamishima, KDD 2003) into ./data/sushi3-2016
set -e
cd "$(dirname "$0")"
mkdir -p data
curl -L -o data/sushi3-2016.zip https://www.kamishima.net/asset/sushi3-2016.zip
unzip -o -q data/sushi3-2016.zip -d data
rm data/sushi3-2016.zip
ls data/sushi3-2016
