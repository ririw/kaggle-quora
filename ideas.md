Ideas
=====

Synthetic data
--------------
 * Reshuffled questions: this seems like it could work, but maybe it's worth
   picking questions that aren't _too_ dissimilar, otherwise they don't teach
   anything
 * Additional data sources?

Features
--------
 * Decompositions of the TFIDF matrix/term matrix
  * SVM
  * NMF

Algorithms
----------
 * LibFFM?
   FFM isn't really applicable here. In an FFM, there are namespaces that are convolved together,
   eg, users and movies, and we learn things from each one from the other, and they fit together.
   We don't have this for words.
 * XTC: expand to more features

Analysis
--------
 * XTC: find the feature importances
