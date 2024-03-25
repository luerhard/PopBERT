library(quanteda)
library(tidyverse)
library(optparse)
library(arrow)
library(popdictR)
library(here)

option_list <- list(
  make_option(c("-f", "--file"), type="character", default=NULL,
              help="dataset file (as parquet)", metavar="character"),
  make_option(c("-o", "--out"), type="character", default=NULL,
              help="output file name ", metavar="character")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)


df <- arrow::read_parquet('/mnt/nvme_storage/git/bert_populism/tmp/gruendl/raw_sents.parquet')

corpus <- quanteda::corpus(df, text_field="sentence")

print("start popdictR...")

result <- popdictR::run_popdict(corpus, return_value="binary")

gruendl_result <- docvars(result) %>%
  select(-c(n_tokens, n_sentences))

print("saving output...")

arrow::write_parquet(gruendl_result, opt$out)

print("done.")
