#!/usr/bin/env R


library("gutenbergr")
library("dplyr")
library("tidytext")
library("stringr")
library("stringi")

works <- gutenberg_works()

author_works <- function(author_name) {
  gutenberg_works() %>%
    filter(author == author_name) %>%
    .[["gutenberg_id"]]
}

lines_to_sentences <- function(lines_data) {
  lines_data$text <- stri_enc_toutf8(lines_data$text, validate=TRUE)
  unnest_tokens(lines_data, sentence, text, token = "sentences")
}

## Downloading and reshaping some of the dataset
ids <- list(
  "keats" = author_works("Whitman, Walt"),
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

## some basic text preprocessing
sentences$sentence <- gsub("[[:punct:]]+","", sentences$sentence)
sentences$sentence <- gsub("\\s+", " ", str_trim(sentences$sentence))
sentences <- sentences %>%
  ungroup() %>%
  filter(!grepl("footnote", sentence)) %>%
  mutate(n_words = str_count(sentence, " ") + 1) %>%
  filter(n_words > 5) %>%
  arrange(desc(n_words))

write.csv(sentences, file = "sentences.csv", row.names = FALSE)
