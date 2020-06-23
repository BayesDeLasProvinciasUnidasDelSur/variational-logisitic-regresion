#!/bin/bash
if [ -d .git ]; then
    a=`grep "$1" .git/config |  wc -l` 
else
    a=`cat .git |  awk -F ': ' '{a=$2"/config" ; print a}' | xargs grep "$1" | wc -l`
fi
if [ $a -gt '0' ]; then
	echo "El remoto $1 ya estaba incluido"
else  
	b=`git ls-remote $1 | wc -l`
	echo $b
	if [ $b -eq 0 ];then
		echo "$1 no parece ser un repositorio"
	else
		git remote set-url --add origin $1
	fi
fi
