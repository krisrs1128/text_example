
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

## Geometric Views: Before Training

AI can often be usefully thought of in geometric terms. Classification is only
possible because one class lives in a different region of a geometric
representation space than another (and when it's impossible, it's because the
classes are all mixed up in the same place).

It's not at all obvious how we can convert something like a sentence into
something geometrically meaningful. We'll reflect on this issue, before diving
more deeply into training classification models.

Historically, one of the first ways people in ML tried geometrically
representing sentences was to convert them into their "bag-of-words". The idea
is to just count the occurrences of the most common (say, 5000 most frequenty)
words. You can then summarize a sentence by a length 5000 vector giving these
frequencies (most of these will be zero), and vectors can be visualized
geometrically.

This is pretty obviously not a satisfying solution -- no one understand a
document by looking at the jumbled up words! An alternative idea is to consider
one-hot encodings of each word. The first word in the vocabulary is (1, 0,
0,...), the second is (0, 1, 0, 0, 0), and so on. You can imagine these words as
being "corners" of the positive half of a very high-dimensional star (try
drawing the representations in the case of a vocabulary of size three: (1, 0,
0), (0, 1, 0), and (0, 0, 1)).


Then, a sentence can be thought of geometrically as a directed curve between the
corners of this star. Visualizing things in very high dimension is tricky, but
the figure below gives some intuition about what we should expect.
