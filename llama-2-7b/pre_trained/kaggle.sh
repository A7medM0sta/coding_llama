# Authenticate with credentials
export KAGGLE_USERNAME="a7med7m0stvfa"
export KAGGLE_KEY="c83d1d29d162caa7e6da0856f8710dbf"

# With Curl
curl -L -o ~/Downloads/model.tar.gz  https://www.kaggle.com/api/v1/models/google/gemma/pyTorch/2b/1/download -u $KAGGLE_USERNAME:$KAGGLE_KEY

# Download specific version (here version 1)
wget https://www.kaggle.com/api/v1/models/google/gemma/pyTorch/2b/1/download --user=$KAGGLE_USERNAME --password=$KAGGLE_KEY --auth-no-challenge