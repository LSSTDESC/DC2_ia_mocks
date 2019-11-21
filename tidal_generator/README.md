# Parallel code for computing the tidal field


This is not a super clean code... but here is how to compile the
different pieces:

So, first of all you need to load the  following modules
```sh
$ module  clear
$ module load python/3.7-anaconda-2019.07 PrgEnv-gnu
$ export MPICC=mpicc
```

```sh
$ cd genericio
$ make py_deps
$ make py_build
$ cp python/*.so ../
```



```sh
$ pip install pmesh
```
