#!/bin/bash
while :
do
	cp ./train/* ./train.bak/ --update
	date
	sleep 60
done
