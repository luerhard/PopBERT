# Hardware

All code except `code/02-create_gbert_model.ipynb` and `code/04-make_predictions.ipynb` was run on an Apple M3 Pro, 18 GB RAM und MacOS 14.4.1

The notebook `code/02-create_gbert_model.ipynb` and `code/04-make_predictions.ipynb` were run on an HPC compute node an AMD EPYC 7513 CPU, 25GB RAM and either an NVIDIA A40 or an NVIDIA A100 GPU.

# Runtime

The runtime for the full code is about 3-5 hours, with package installations, `04-make_predictions.ipynb` and `08-run_gruendl.R` taking up the large majority of the compute time.

# Software Versions

The replication repository uses R 4.3.3 and Python 3.9, although other versions might work.

## R Package versions

All R package versions are pinned and the environment is managed using the `renv`package.

All installed packages are listed in the file: `renv.lock`.

To install all R dependencies, use:

```
Rscript -e "source("renv/activate.R); renv::restore()"
```

from the root project folder.

## Python Package versions

All Python package versions are pinned and the environment is managed using the `poetry` package.

All installed packages are listed in the file: `poetry.lock`.

To install all Python dependencies, use: 

```
poetry install
```

from the root project folder.

## Linux dependencies

If run from a Ubuntu 22.04 base image on Code Ocean, the following apt-installable packages are missing:

```
gfortran
liblapack-dev
libopenblas-dev
libxml2-dev
libfontconfig1-dev
libharfbuzz-dev
libfribidi-dev
libfreetype6-dev
libpng-dev
libtiff5-dev
libjpeg-dev
libbz2-dev
```

# Replication of all stages



## stage dependency graph

```

+-----------------------+  
| annotator_performance |  
+-----------------------+  
+----------------------+ 
| create_popbert_model | 
+----------------------+ 
+-------------------+  
| model_performance |  
+-------------------+  
                                                                             +-------------------------------------+                                                                                  
                                                                           **| data/raw/sentences.parquet.gzip.dvc |***                                                                               
                                                             ************** *+-------------------------------------+** **************                                                                 
                                              ***************      *********            **               **           *********      ***************                                                  
                               ***************            *********                  ***                   ***                 ********             **************                                    
                 **************                   ********                         **                         **                       ********                   **************                      
         ********                            *****                    +------------------+                 +-------------+                     *****                            ********              
         *                                     ***              ******| make_predictions |***********      | run_gruendl |                 ******                                      *              
         *                                        **************      +------------------+*****      ******+-------------+            *****                                            *              
         *                            *************  ***               ***                     *****              *     *********************                                          *              
         *               *************                  ***         ***                             ******        *       ******             *******************                       *              
         *        *******                                  **     **                                      ***     *    ***                                      ***********            *              
+----------------+                                  +--------------------+                             +-------------------+                                              +------------------------+  
| all_dimensions |                                  | populism_by_speech |                             | selected_examples |                                              | populism_by_politician |  
+----------------+                                  +--------------------+                             +-------------------+                                              +------------------------+  
```