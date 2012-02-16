#!/bin/sh

set -o verbose

mencoder "mf://*.png" -mf type=png:fps=3 -ovc lavc -o output.avi

