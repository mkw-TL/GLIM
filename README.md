## GLIM
Inferential Models allow the user to acheive Bayesian style posteriors, with no priors, and with type 1 error control.

This package implements Ryan Martin's IM framework within the GLM context. Computations primarily based on ideas from: https://arxiv.org/abs/2501.10585

## To install:
In your console, run this line:
remotes::install_github("mkw-TL/GLIM", ref = "alpha1")

You will need devtools and Rtools in order to successfully build this on Windows

In the source files, there is an Example_script.R which will show you how to run the program.

Linux users are on their own as always. You may need to:
sudo apt-get install libomp-dev
