Ideas
=====

Synthetic data
--------------
 * Reshuffled questions: this seems like it could work, but maybe it's worth
   picking questions that aren't _too_ dissimilar, otherwise they don't teach
   anything
 * Additional data sources?
   * http://www.cis.upenn.edu/~ccb/ppdb/ -- new data

Features
--------
 * Decompositions of the TFIDF matrix/term matrix
  * DONE SVM
  * NMF
 * DONE http://proceedings.mlr.press/v37/kusnerb15.pdf
 * DONE https://openreview.net/pdf?id=SyK00v5xx (but not perfected)
 * https://datawhatnow.com/simhash-question-deduplicatoin/

Algorithms
----------
 * LibFFM?
   FFM isn't really applicable here. In an FFM, there are namespaces that are convolved together,
   eg, users and movies, and we learn things from each one from the other, and they fit together.
   We don't have this for words.
 * DONE XTC: expand to more features
  * This worked :)
 * doc2vec
 

Analysis
--------
 * DONE XTC: find the feature importances
