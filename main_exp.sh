#!/usr/bin/bash

files=$(ls | grep -E '_testing_')

for file in $files
do
	$(bash $file)
done
