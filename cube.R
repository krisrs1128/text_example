
library("ggplot2")
library("readr")
library("dplyr")
library("purrr")
library("igraph")
library("tidytext")

repl_sent <- function(x, vocab) {
  y <- strsplit(x, " ")[[1]]
  y[!(y %in% vocab)] <- "UNK"
  paste(y, collapse = " ")
}

sentence_graph <- function(x) {
  y <- strsplit(x, " ")[[1]]
  which(y %in% vocab)
  edge_list <- list()
  for (i in seq_along(y)) {
    if (i == length(y)) break
    edge_list[[i]] <- c(y[i], y[i + 1])
  }

  do.call(rbind, edge_list)
}

sentences <- read_csv("data_prep/sentences.csv") %>%
  mutate(sentence_id = row_number())

# keep only the 250 most common words
words <- sentences %>%
  unnest_tokens(word, sentence)

word_counts <- words %>%
  group_by(word) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

vocab <- word_counts[1:2000, ]$word

sentences <- sentences %>%
  group_by(sentence) %>%
  mutate(replaced = repl_sent(sentence, vocab))

G <- sentences$replaced[1000] %>%
  sentence_graph() %>%
  graph.edgelist()

ggplot(G,
       layout = "circle",
       aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_edges(alpha = 0.1, width = 0.01, color = "grey50", arrow = arrow(length = unit(2, "pt"), type = "closed")) +
  geom_text(aes(label = vertex.names), size = 3) +
  theme_blank()
