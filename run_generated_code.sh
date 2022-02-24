#!/bin/bash

g++ -I runtime generated_code/$1.cpp -g -o generated_code/$1 && ./generated_code/$1
