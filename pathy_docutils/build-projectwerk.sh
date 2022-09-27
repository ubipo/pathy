#!/bin/env bash
##############################################
# build-projectwerk.sh                       #
#                                            #
# Is your name Pieter? If no, don't use this #
##############################################
set -e
shopt -s dotglob

USERNAME=$1

echo -n "Password: " 
read -s PASSWORD
echo

TEMP_DIR="/tmp/cv_drone_build_projectwerk"
rm -rf $TEMP_DIR
mkdir $TEMP_DIR

REPO_DIR="$TEMP_DIR/repo"
git clone "https://$USERNAME:$PASSWORD@projektwerk.ucll.be/git/cv_drone" $REPO_DIR
cd $REPO_DIR

python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

export PYTHONPATH=$PYTHONPATH:$REPO_DIR/pathy_docutils/
jupyter-book build .
cp -r _build $TEMP_DIR

git checkout --orphan docs
git rm -rf .
git clean -fxd

mv $TEMP_DIR/_build/html/* ./

git add -A
git commit -m "Automatic docs build $(date)"
git push origin --delete docs
git push origin docs
