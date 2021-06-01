# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 17:51:23 2021

@author: athir
"""
import csv
import random

edited_data = []

with open('speech_blocks_objects.txt') as speechblocks:
    speechblocks_objects = speechblocks.read().splitlines() 

with open('train100k.txt') as training_data:
    data_reader = csv.reader(training_data, delimiter='\t')

    for line in data_reader:
        if line[0] == 'CapableOf':
            new_line = []
            new_line.append(line[0])
            new_line.append(line[1])
            
            max_scene_length = random.randint(0, 5)
            
            object_list = []
            for i in speechblocks_objects:
                if i + ' ' in line[2] or line[2].endswith(i):
                    object_list.append(i)
            while(len(object_list) < max_scene_length):
                random_number = random.randint(0, 733)
                if speechblocks_objects[random_number] not in object_list:
                    object_list.append(speechblocks_objects[random_number])
            random.shuffle(object_list)
            object_list_str = ''
            for i in range(0, len(object_list)):
                if i == len(object_list) - 1:
                    object_list_str = object_list_str + object_list[i]
                else:
                    object_list_str = object_list_str + object_list[i] + ' '
            new_line.append(object_list_str)
            new_line.append(line[2])
            new_line.append(line[3])
            
            edited_data.append(new_line)


with open('new_training_data.txt', 'w') as f:
    for l in edited_data:
        f.write('\t'.join(l[0:]) + '\n')