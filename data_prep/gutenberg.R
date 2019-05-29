#!/usr/bin/env R
#'
#' Data from Project Gutenberg
#'
#' This is an example of using the gutenbergr to download and lightly preprocess
#' sentences for later use in a classification system. The question here is to
#' classify between poems by john keats and william wordsworth -- they are both
#' poets with quite a few poems on Project Gutenberg.
library("gutenbergr")
library("dplyr")
library("tidytext")
library("stringr")
library("stringi")

# Simple wrapper functions

#' Gutenberg IDs for an Author
author_works <- function(author_name) {
  gutenberg_works() %>%
    filter(author == author_name) %>%
    .[["gutenberg_id"]]
}

#' Use tidytext to split sentences
lines_to_sentences <- function(lines_data) {
  lines_data$text <- stri_enc_toutf8(lines_data$text, validate=TRUE)
  unnest_tokens(lines_data, sentence, text, token = "sentences")
}

#' Downloading and reshaping some of the dataset
works <- gutenberg_works()
ids <- list(
  "keats" = author_works("Keats, John"),
  "wordsworth" = author_works("Wordsworth, William")
)

texts <- list(
  "keats" = gutenberg_download(ids[["keats"]], verbose=TRUE),
  "wordsworth" = gutenberg_download(ids[["wordsworth"]])
)

sentences <- rbind(
  data.frame(author = "keats", lines_to_sentences(texts[["keats"]])),
  data.frame(author = "wordsworth", lines_to_sentences(texts[["wordsworth"]]))
)

#' some basic text preprocessing
sentences$sentence <- gsub("[[:punct:]]+","", sentences$sentence)
sentences$sentence <- gsub("\\s+", " ", str_trim(sentences$sentence))
sentences <- sentences %>%
  ungroup() %>%
  filter(!grepl("footnote", sentence)) %>%
  mutate(n_words = str_count(sentence, " ") + 1) %>%
  filter(n_words > 5) %>%
  arrange(desc(n_words))

write.csv(sentences, file = "sentences.csv", row.names = FALSE)
