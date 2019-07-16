#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 22:55:04 2019

@author: jiahui
"""

import os
from PIL import Image
cade=os.listdir('/home/jiahui/Desktop/instb/')
print(cade)
for i in cade:
    cate=os.listdir('/home/jiahui/Desktop/instb/'+i+'/')
    for j in cate:
        foo=Image.open('/home/jiahui/Desktop/instb/'+i+'/'+j)
        (h,w)=foo.size
        foo=foo.resize((int(h/4),int(w/4)),Image.ANTIALIAS)
        foo.save('/home/jiahui/Desktop/instb/'+i+'/sca'+j,optimize=True,quality=95)