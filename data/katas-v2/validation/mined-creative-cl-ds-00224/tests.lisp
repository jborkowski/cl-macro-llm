;;; tests.lisp — generated from j14i/cl-ds row 224
;;; instruction: Write a Common Lisp macro `assert-non-empty` that evaluates a sequence expression once, signals an error if the sequence
;;; category: validation | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((assert-non-empty items) . (LET ((#:G1 ITEMS)) (WHEN (ZEROP (LENGTH #:G1)) (ERROR "Sequence is empty")) #:G1)))
