library(quanteda)
library(tidyverse)
library(arrow)
library(popdictR)
library(here)

input_path <- here("data", "raw", "sentences.parquet.gzip")
out_path <- here("data", "interim", "gruendl.parquet.gzip")

df <- arrow::read_parquet(input_path)

corpus <- quanteda::corpus(df, text_field="text")

print("start popdictR...")

result <- popdictR::run_popdict(corpus, return_value="binary")

gruendl_result <- docvars(result) %>%
    rename(gruendl = dict_gruendl_2020) %>% 
    select(sample_id, gruendl)

print("saving output...")

arrow::write_parquet(gruendl_result, out_path, compression="gzip")

print("done.")
