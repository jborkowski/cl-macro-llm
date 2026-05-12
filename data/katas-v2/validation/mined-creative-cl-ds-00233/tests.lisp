;;; tests.lisp — generated from j14i/cl-ds row 233
;;; instruction: Write a Common Lisp macro `assert-keyword` that signals an error if its argument is not a keyword; otherwise returns the
;;; category: validation | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((assert-keyword mode) . (LET ((#:G1 MODE)) (UNLESS (KEYWORDP #:G1) (ERROR "Expected a keyword, got ~S" #:G1)) #:G1)))
