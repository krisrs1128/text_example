
# A Text Classification Primer (on Mila Cluster)

Today, some ML for poets.

In this primer, we will
* Download some poems by Keats and Donne
* Look at some embeddings of sentences of these poems
* Write a simple classifier, to distinguish between the authors
* Inspect the results, to hone in on areas worth improving

... all on the Mila cluster.

The hope is that you'll be able to actually follow along with this document.
It's deliberately not written as a notebook, because (1) we'll be switching
between the terminal and different languages, and (2) I want to give an idea of
how an interconnected ML pipeline, as opposed to a one-off script.

## Running this code

All the software dependencies for this project are contained in this singularity
recipe. To run it in an interactive session on the Mila cluster, follow these
steps.

1) Download the singularity image
2) Transfer it to the cluster
3) Launch an interactive session
4) Startup a terminal in the singularity image

## Preparing the Data

Data collection and preprocessing is often the trickiest part of doing ML in
practice, but it's almost universally overlooked in coursework and textbooks,
partly because it's so context dependent. For this example, let's download poems
from Project Gutenberg, and then extract sentences that will make up our
eventual classification dataset.

Project Gutenberg is a pretty popular website among data scientists, so there's
actually an R library for interfacing with it. 

The script `sentences.R` will download all the works by these authors from
Gutenberg, extract all the sentences, do some preprocessing, and write them to
file scalled `sentences.csv`.

(give a little more detail about the functions)

We can run it from the command line using `R sentences.R` -- in a pipeline, this
would be put in some shell script.
