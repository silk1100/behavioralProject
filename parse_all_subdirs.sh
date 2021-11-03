#!/usr/bin/bash
fldrs=$(ls)

declare -a output

function recursive_search()
{
	if [ -d $1 ];then
		allsubdirs=$(ls $1)
		for subdir in ${allsubdirs[@]}
		do
			recursive_search $1/$subdir
		done
	else
		output+=($(dirname $1))
	fi
	echo ${output[@]}
}


for i in ${fldrs[@]}
do
	if [ -d $i ]; then
		finalout+=($(recursive_search $i))
	fi

done

declare -a cleanout

for i in ${finalout[@]}
do
	found=0
	for j in ${cleanout[@]}
	do
		if [[ $i == $j ]]; then
			found=1
			break
		fi
	done
	if (($found == 0)); then
		cleanout+=($i)
	fi

done

echo ${#finalout[@]}
echo ${#cleanout[@]}
echo ${cleanout[@]}
