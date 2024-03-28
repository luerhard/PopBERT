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
libcurl4-openssl-dev
```

# Replication of results

## Manual Replication of the results

Please note that the tables are printed as `LaTeX` code which we manually edited to adjust the look of the table where necessary.

## Table 1: Number of annotated sentences in the dataset

To recreate this table, run `code/01-annotator_performance.ipynb`.
It reads the data from the file `data/labeled_data/full.csv.zip`, and saves the result as `results/tables/coder_agreement.tex`.

## The Model

Please note here: To our knowledge, it is virtually impossible to create BERT Transformer models that are 100% replicable. This is not only dependent on different seeds, the transformer and the PyTorch version, but replicability is also not given across different CUDA versions, GPU and even CPU models. We are therefore taking a compromise approach here. 

The model we created with the script `02-create_popbert_model.ipynb` is published version-controlled on huggingface and is used with the commit hash from number 03 in the notebook series under `code/`. 

If you want to use the model you have calculated, replace the following code block:

```
COMMIT_HASH = "cf44004e90045cde298e28605ff105747d58aa7a"

tokenizer = AutoTokenizer.from_pretrained("luerhard/PopBERT", revision=COMMIT_HASH)
model = AutoModelForSequenceClassification.from_pretrained(
    "luerhard/PopBERT", revision=COMMIT_HASH
).to(DEVICE)
```

with

```
tokenizer = AutoTokenizer.from_pretrained("luerhard/PopBERT", revision=COMMIT_HASH)
model = AutoModelForSequenceClassification.from_pretrained(
  str(str.PATH / "results/popbert_model"), use_local_files_only=True
).to(DEVICE)
```

This replacement is possible in the following notebooks:

- `code/03-model_performance.ipynb`
- `code/04-make_predictions.ipynb`

Note, however, that the subsequent results are similar but not necessarily identical to the ones, published in the main article.

## Table 2: Performance of the model on the 20% test set

To recreate this table, run `code/03-model_performance.ipynb`.
It reads data from the file `data/labeled_data/test.csv.zip` and saves the result to `results/tables/model_performance.tex`.

## Replication using Data Version Control (DVC)

### stage dependency graph

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
