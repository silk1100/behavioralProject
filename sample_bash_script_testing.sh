#!/usr/bin/bash
declare -a output

#######################################
# Recursively iterate over all subdirectories of a passed directory
# Globals:
#   output
# Arguments:
#   Folder as a base/starting point
#######################################
function search_dirs () {
    # $1: is the main_path
    subdirs="$(ls $1)"
    for sub in $subdirs
    do
        fullsubdir="$1/$sub"
        if [[ -d $fullsubdir ]]; then
            echo "$(search_dirs $fullsubdir)"
        else
            output+=("$(dirname $fullsubdir)")
            # echo "$(dirname $fullsubdir | uniq -u)"
        fi
    done

    echo $output
}

fldrs="$(search_dirs ./experiments_testing)"
cnt=1
for fld in $fldrs
do
    stats="$(echo "#!/usr/bin/bash
cd /home/tarek/PhD/behavioral_project/src
python ./main.py -i .$fld -o ../output" > sample_testing_$cnt.sh)"
    ((cnt+=1))
done
