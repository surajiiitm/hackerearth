#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:50:01 2017

@author: suraj
"""

#import flask library
from flask import Flask

#initisalise tha app from flask
app = Flask(__name__)


#define a route to hello world function
@app.route('/')
def hello_world():
    return 'Hello World!'

#run the app on localhost 8085
app.run(debug=True, port = 8085)