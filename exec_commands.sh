#!/bin/bash

files=$(ls . | egrep '[a-z]*_[0-9]*.sh')
for file in $files
do
    $("bash ./$file")
done


