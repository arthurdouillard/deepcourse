#!/bin/bash

set -e

/usr/local/bin/hugo-0.79.0/hugo
git add docs/
git add -u
git commit -m "Update website"
git push origin master
