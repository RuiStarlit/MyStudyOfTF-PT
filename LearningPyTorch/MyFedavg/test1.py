# -*- coding:utf-8 -*-
"""
Author: RuiStarlit
File: test1
Project: LearningPyTorch
Create Time: 2021-07-07

"""
import os


if __name__ =='__main__':
    f = open('fed_{}_log.txt'.
            format('mnist'), 'a+')
    f.write('This  is my test log' + '\n')
    f.write('The second line'+ '\n')
    f.close()